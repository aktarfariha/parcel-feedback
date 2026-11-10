"""Parcel feedback system - backend.

Data layout (put your Colab/Drive output in ./data):
    data/<name>.jpg              original image
    data/<name>_parcels.geojson  parcel polygons in pixel coordinates

Run:  uvicorn app.main:app --reload
"""
import json
import math
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import tiles
from .learner import Learner

ROOT = Path(__file__).resolve().parent.parent
# The three locations can be moved with environment variables (useful on a server or for tests).
DATA_DIR = Path(os.environ.get("PARCEL_DATA_DIR") or ROOT / "data")
DB_PATH = Path(os.environ.get("PARCEL_DB") or ROOT / "feedback.db")
MODEL_DIR = Path(os.environ.get("PARCEL_MODEL_DIR") or ROOT / "model")
STATIC_DIR = ROOT / "static"
SETTINGS_PATH = ROOT / "settings.json"
DATA_DIR.mkdir(exist_ok=True)

LABELS = ["crop_field", "pond", "house", "road", "tree_cover", "water", "other"]
BOUNDARY_ISSUES = ["ok", "should_split", "should_merge", "wrong_edge", "not_a_parcel"]

app = FastAPI(title="Parcel Feedback")


# ---------------------------------------------------------------- database
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                weight REAL NOT NULL DEFAULT 1.0
            );
            CREATE TABLE IF NOT EXISTS label_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image TEXT NOT NULL, parcel_id TEXT NOT NULL,
                user_id TEXT NOT NULL, label TEXT NOT NULL, ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS boundary_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image TEXT NOT NULL, parcel_id TEXT NOT NULL,
                user_id TEXT NOT NULL, issue TEXT NOT NULL,
                polygon TEXT, ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS missed_parcels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image TEXT NOT NULL, user_id TEXT NOT NULL,
                polygon TEXT NOT NULL, label TEXT, ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_label ON label_feedback(image, parcel_id);
            CREATE INDEX IF NOT EXISTS ix_bnd ON boundary_feedback(image, parcel_id);
            """
        )
        # Older databases (v1/v2) lack the columns that record what the system guessed when a
        # person answered. Add them without touching existing data.
        have = {r["name"] for r in con.execute("PRAGMA table_info(label_feedback)")}
        for col, typ in (("pred_label", "TEXT"), ("pred_conf", "REAL"), ("model_version", "INTEGER"), ("saw_guess", "INTEGER")):
            if col not in have:
                con.execute(f"ALTER TABLE label_feedback ADD COLUMN {col} {typ}")


init_db()
learner = Learner(DATA_DIR, MODEL_DIR, db, LABELS)
learner.init_tables()
learner.startup()  # loads the saved model, or trains one in the background if enough labels exist


# ------------------------------------------------------------------ helpers
def list_images():
    return sorted(p.name[: -len("_parcels.geojson")] for p in DATA_DIR.glob("*_parcels.geojson"))


def load_geojson(name: str):
    path = DATA_DIR / f"{name}_parcels.geojson"
    if not path.exists():
        raise HTTPException(404, f"No parcel file for '{name}'")
    return json.loads(path.read_text())


def consensus(votes):
    """votes: list of (label, weight). Returns (label, confidence, n)."""
    if not votes:
        return None, 0.0, 0
    tally = {}
    for label, w in votes:
        tally[label] = tally.get(label, 0.0) + w
    total = sum(tally.values())
    best = max(tally, key=tally.get)
    return best, tally[best] / total, len(votes)


def entropy(votes):
    if not votes:
        return 99.0  # unseen parcels are the most informative
    tally = {}
    for label, w in votes:
        tally[label] = tally.get(label, 0.0) + w
    total = sum(tally.values())
    return -sum((v / total) * math.log(v / total + 1e-12) for v in tally.values())


def votes_for(con, image):
    rows = con.execute(
        """SELECT f.parcel_id, f.label, COALESCE(u.weight, 1.0) AS w
           FROM label_feedback f LEFT JOIN users u ON u.user_id = f.user_id
           WHERE f.image = ?""",
        (image,),
    ).fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["parcel_id"], []).append((r["label"], r["w"]))
    return out


# --------------------------------------------------------------------- API
@app.get("/api/config")
def config():
    return {
        "labels": LABELS,
        "boundary_issues": BOUNDARY_ISSUES,
        "tile_url": tiles.TILE_URL,
        "tile_attribution": tiles.ATTRIBUTION,
        "max_zoom": tiles.MAX_ZOOM,
        "max_side_px": tiles.MAX_SIDE_PX,
    }


# ---------------------------------------------------------- detector connection
QUALITY = {"fast": (32, 1), "accurate": (48, 2)}  # (points_per_side, crop_n_layers)


def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}


def save_settings(d):
    SETTINGS_PATH.write_text(json.dumps(d))


def parse_link(link: str):
    """'https://abc.trycloudflare.com?token=XYZ' -> ('https://abc.trycloudflare.com', 'XYZ')"""
    link = (link or "").strip()
    if not link:
        return "", ""
    if "://" not in link:
        link = "https://" + link
    u = urlparse(link)
    token = parse_qs(u.query).get("token", [""])[0]
    return f"{u.scheme}://{u.netloc}{u.path.rstrip('/')}", token


def detector_conn():
    return parse_link(os.environ.get("DETECTOR_URL") or load_settings().get("detector_link", ""))


def detector_status():
    base, token = detector_conn()
    if not base:
        return {"configured": False, "connected": False,
                "message": "No detector connected yet. Paste the connection link from the Colab notebook."}
    try:
        r = httpx.get(f"{base}/health", headers={"X-Token": token}, timeout=8)
    except httpx.HTTPError:
        return {"configured": True, "connected": False,
                "message": "Cannot reach the detector. Is the Colab notebook still running? "
                           "Colab gives a new link every time you start it."}
    if r.status_code == 401:
        return {"configured": True, "connected": False,
                "message": "The detector rejected the token. Paste the complete connection link again."}
    if r.status_code != 200:
        return {"configured": True, "connected": False, "message": f"Detector answered HTTP {r.status_code}."}
    h = r.json()
    if h.get("error"):
        return {"configured": True, "connected": False, "message": "Detector model failed to load: " + h["error"]}
    if not h.get("model_loaded"):
        return {"configured": True, "connected": False, "message": "Detector is still loading the model. Wait a minute."}
    return {"configured": True, "connected": True, "device": h.get("device"), "queue": h.get("queue", 0),
            "message": f"Connected ({h.get('device')})"}


class LinkIn(BaseModel):
    link: str


@app.get("/api/detector/status")
def api_detector_status():
    return detector_status()


@app.post("/api/detector/link")
def api_detector_link(body: LinkIn):
    s = load_settings()
    s["detector_link"] = body.link.strip()
    save_settings(s)
    return detector_status()


# ------------------------------------------------------------ add a new map area
@app.get("/api/plan")
def api_plan(west: float, south: float, east: float, north: float):
    try:
        return tiles.plan_area([west, south, east, north])
    except tiles.AreaError as e:
        raise HTTPException(400, str(e))


site_jobs: dict = {}


class SiteIn(BaseModel):
    bbox: list
    name: str | None = None
    quality: str = "fast"


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:30] or "area"


def run_site_job(job_id: str, plan: dict, site: str, quality: str):
    job = site_jobs[job_id]

    def say(msg):
        job["message"] = msg

    img_path = DATA_DIR / f"{site}.jpg"
    try:
        base, token = detector_conn()
        st = detector_status()
        if not st["connected"]:
            raise RuntimeError(st["message"])
        hdr = {"X-Token": token}

        say(f"Downloading {plan['tiles']} satellite tiles...")
        img, warnings = tiles.fetch_mosaic(plan)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_path.write_bytes(jpg.tobytes())
        job["warnings"] = warnings

        pts, layers = QUALITY.get(quality, QUALITY["fast"])
        say("Sending the image to the detector...")
        with httpx.Client(timeout=60, headers=hdr) as c:
            r = c.post(f"{base}/jobs", content=jpg.tobytes(), headers={"Content-Type": "image/jpeg"},
                       params={"max_dim": 1024, "points_per_side": pts, "crop_n_layers": layers})
            if r.status_code != 200:
                raise RuntimeError(f"Detector refused the job: {r.text[:200]}")
            remote_id = r.json()["job_id"]

            fails, deadline, result = 0, time.time() + 45 * 60, None
            while time.time() < deadline:
                time.sleep(3)
                try:
                    pr = c.get(f"{base}/jobs/{remote_id}")
                    fails = 0
                except httpx.HTTPError:
                    fails += 1
                    if fails >= 10:
                        raise RuntimeError("Lost contact with the detector. Check that the Colab notebook is still running.")
                    continue
                if pr.status_code == 404:
                    raise RuntimeError("The detector restarted and lost this job. Start it again and retry.")
                d = pr.json()
                if d["status"] == "queued":
                    say(f"Waiting in the detector queue (position {d.get('queue_position', '?')})...")
                elif d["status"] == "running":
                    say(f"Detecting parcels on the GPU... {int(d['elapsed'])}s (usually 1-5 minutes)")
                elif d["status"] == "error":
                    raise RuntimeError("Detector error: " + d["error"])
                elif d["status"] == "done":
                    result = d["result"]
                    break
            if result is None:
                raise RuntimeError("Detection took longer than 45 minutes and was stopped.")

        say("Saving parcels...")
        features = []
        for i, ring in enumerate(result["polygons"]):
            if len(ring) < 4:
                continue
            area = 0.5 * abs(sum(ring[k][0] * ring[k + 1][1] - ring[k + 1][0] * ring[k][1] for k in range(len(ring) - 1)))
            features.append({"type": "Feature", "id": i,
                             "properties": {"parcel_id": f"{site}_{i}", "area_px": round(area, 1), "label": None},
                             "geometry": {"type": "Polygon", "coordinates": [ring]}})
        gj = {"type": "FeatureCollection", "image": site, "width": plan["width"], "height": plan["height"],
              "georef": tiles.georef_from_plan(plan), "quality": quality, "features": features}
        # the geojson is written last: its presence marks the site as ready
        (DATA_DIR / f"{site}_parcels.geojson").write_text(json.dumps(gj))
        job.update(status="done", site=site, message=f"Done: {len(features)} parcels", n_parcels=len(features))
    except Exception as e:
        img_path.unlink(missing_ok=True)
        job.update(status="error", error=str(e) or type(e).__name__, message="Failed")


@app.post("/api/sites")
def create_site(body: SiteIn):
    try:
        plan = tiles.plan_area(body.bbox)
    except tiles.AreaError as e:
        raise HTTPException(400, str(e))
    st = detector_status()
    if not st["connected"]:
        raise HTTPException(409, st["message"])
    site = f"{slugify(body.name)}_{time.strftime('%m%d-%H%M%S')}"
    job_id = uuid.uuid4().hex[:10]
    site_jobs[job_id] = {"status": "running", "message": "Starting...", "site": None, "started": time.time(),
                         "plan": {k: plan[k] for k in ("zoom", "width", "height", "m_per_px", "tiles")}}
    threading.Thread(target=run_site_job, args=(job_id, plan, site, body.quality), daemon=True).start()
    return {"job_id": job_id, "site": site}


@app.get("/api/jobs/{job_id}")
def get_site_job(job_id: str):
    job = site_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return {**job, "elapsed": round(time.time() - job["started"])}


@app.get("/api/images/{name}/export.geojson")
def export_geojson(name: str):
    """Parcels + current consensus labels as standard GeoJSON in lon/lat (opens in QGIS etc.)."""
    gj = parcels(name)
    g = gj.get("georef")
    if not g:
        raise HTTPException(404, "This image has no map location. Only areas created with 'Add new area' can be exported.")

    def conv(ring):
        return [list(tiles.pixel_to_lonlat(x, y, g)) for x, y in ring]

    m2_per_px2 = g["m_per_px"] ** 2
    feats = []
    for f in gj["features"]:
        pr = f["properties"]
        feats.append({"type": "Feature", "properties": {
            "parcel_id": pr["parcel_id"], "kind": "detected", "label": pr.get("consensus_label"),
            "confidence": pr.get("confidence"), "n_votes": pr.get("n_votes"),
            "predicted_label": pr.get("predicted_label"), "predicted_confidence": pr.get("predicted_confidence"),
            "outline_edited": bool(pr.get("edited")),
            "boundary_issues": pr.get("boundary_issues"), "area_m2": round(pr["area_px"] * m2_per_px2, 1)},
            "geometry": {"type": "Polygon", "coordinates": [conv(f["geometry"]["coordinates"][0])]}})
    for m in gj.get("missed_parcels", []):
        ring = m["polygon"] + [m["polygon"][0]]
        area_px = 0.5 * abs(sum(ring[k][0] * ring[k + 1][1] - ring[k + 1][0] * ring[k][1] for k in range(len(ring) - 1)))
        feats.append({"type": "Feature", "properties": {"parcel_id": f"user_added_{m['id']}", "kind": "user_added",
                                                         "label": m.get("label"), "area_m2": round(area_px * m2_per_px2, 1)},
                      "geometry": {"type": "Polygon", "coordinates": [conv(ring)]}})
    return JSONResponse({"type": "FeatureCollection", "name": name, "features": feats},
                        headers={"Content-Disposition": f'attachment; filename="{name}.geojson"'})


@app.get("/api/images")
def images():
    return list_images()


@app.get("/api/images/{name}/file")
def image_file(name: str):
    for ext in ("jpg", "jpeg", "png"):
        p = DATA_DIR / f"{name}.{ext}"
        if p.exists():
            return FileResponse(p)
    raise HTTPException(404, "Image not found")


@app.get("/api/images/{name}/parcels")
def parcels(name: str, user_id: str = ""):
    gj = load_geojson(name)
    try:
        guesses = learner.predict_image(name) or {}
        gj["model_version"] = learner.bundle["version"] if learner.bundle else None
    except Exception as e:  # the page must still work if the learner has a problem
        guesses, gj["model_version"], gj["learner_error"] = {}, None, f"{type(e).__name__}: {e}"
    with db() as con:
        mine = {r["parcel_id"]: r["label"] for r in con.execute(
            "SELECT parcel_id, label FROM label_feedback WHERE image=? AND user_id=?", (name, user_id))} if user_id else {}
        votes = votes_for(con, name)
        issues = {}
        for r in con.execute(
            "SELECT parcel_id, issue, COUNT(*) c FROM boundary_feedback WHERE image=? GROUP BY parcel_id, issue",
            (name,),
        ):
            issues.setdefault(r["parcel_id"], {})[r["issue"]] = r["c"]
        missed = [
            {"id": r["id"], "polygon": json.loads(r["polygon"]), "label": r["label"]}
            for r in con.execute("SELECT * FROM missed_parcels WHERE image=?", (name,))
        ]
    ov = learner.overrides(name)  # outlines corrected by people with the Redraw tool
    for feat in gj["features"]:
        pid = feat["properties"]["parcel_id"]
        if pid in ov:
            ring = ov[pid] + [ov[pid][0]]
            feat["properties"]["original_geometry"] = feat["geometry"]
            feat["geometry"] = {"type": "Polygon", "coordinates": [ring]}
            feat["properties"]["area_px"] = round(poly_area(ring), 1)
            feat["properties"]["edited"] = True
        label, conf, n = consensus(votes.get(pid, []))
        g = guesses.get(pid)
        feat["properties"].update(
            consensus_label=label, confidence=round(conf, 3), n_votes=n,
            boundary_issues=issues.get(pid, {}), my_label=mine.get(pid),
            predicted_label=g["label"] if g else None,
            predicted_confidence=g["conf"] if g else None,
            predicted_top=g["top"] if g else [],
            predicted_seen=g["oob"] if g else False,  # True = the system's answer for a parcel people already labelled
        )
    gj["missed_parcels"] = missed
    return gj


@app.get("/api/images/{name}/next")
def next_parcel(name: str, user_id: str = "", exclude: str = ""):
    """Active learning: ask about the parcel we are least sure of.
    Skips parcels this user already answered, and any the person chose to skip (exclude=id1,id2,...)."""
    gj = load_geojson(name)
    skipped = {s for s in exclude.split(",") if s}
    # how common is each label in the training data? Rare labels get a bonus, so the system
    # asks "is this a pond?" often enough to collect examples of the types it has seen least.
    freq = {}
    if learner.bundle:
        cc = learner.bundle["metrics"].get("class_counts", {})
        tot = sum(cc.values()) or 1
        freq = {k: v / tot for k, v in cc.items()}
    with db() as con:
        votes = votes_for(con, name)
        mine = {
            r["parcel_id"]
            for r in con.execute(
                "SELECT parcel_id FROM label_feedback WHERE image=? AND user_id=?", (name, user_id)
            )
        }
    try:
        guesses = learner.predict_image(name) or {}
    except Exception:
        guesses = {}
    cands = []
    for feat in gj["features"]:
        pid = feat["properties"]["parcel_id"]
        if pid in mine or pid in skipped:
            continue
        v, g = votes.get(pid, []), guesses.get(pid)
        if not v:
            # nobody has answered: ask where the model is most unsure (entropy of its guess),
            # with a bonus when it guesses a label it has rarely seen
            score = 2.0 + (g["entropy"] + 0.7 * (1 - freq.get(g["label"], 0.0)) if g else 0.0)
        else:
            # already answered: ask again only where people disagree or the model disagrees with them
            score = entropy(v)
            if g and consensus(v)[0] != g["label"]:
                score += 0.5
        cands.append((score + random.random() * 1e-3, pid))
    if not cands:
        return {"parcel_id": None}
    cands.sort(reverse=True)
    return {"parcel_id": cands[0][1]}


class LabelIn(BaseModel):
    image: str
    parcel_id: str
    user_id: str
    label: str
    saw_guess: bool = False  # did the person see the system's guess before answering?


class BoundaryIn(BaseModel):
    image: str
    parcel_id: str
    user_id: str
    issue: str
    polygon: list | None = None  # optional user-redrawn polygon [[x,y],...]


class MissedIn(BaseModel):
    image: str
    user_id: str
    polygon: list
    label: str | None = None


def ensure_user(con, user_id):
    if not user_id.strip():
        raise HTTPException(400, "user_id required")
    con.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))


# ------------------------------------------------------- redraw a wrong boundary
def poly_area(ring):
    """Area of a polygon given as a closed ring [[x, y], ..., first point again]."""
    return 0.5 * abs(sum(ring[k][0] * ring[k + 1][1] - ring[k + 1][0] * ring[k][1] for k in range(len(ring) - 1)))


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def self_intersects(pts):
    """True when two non-neighbouring edges of the polygon cross (a 'bow-tie')."""
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # first and last edge touch at the start point
            c, d = pts[j], pts[(j + 1) % n]
            if _ccw(a, b, c) * _ccw(a, b, d) < 0 and _ccw(c, d, a) * _ccw(c, d, b) < 0:
                return True
    return False


def clean_polygon(points, width, height):
    """Validate an outline drawn by a person. Returns an open ring of [x, y] or raises a friendly 400."""
    if not isinstance(points, list) or not (3 <= len(points) <= 500):
        raise HTTPException(400, "The outline needs at least 3 points.")
    out = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except Exception:
            raise HTTPException(400, "The outline contains an invalid point.")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise HTTPException(400, "The outline contains an invalid point.")
        x, y = min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))  # keep inside the picture
        if not out or abs(out[-1][0] - x) > 0.5 or abs(out[-1][1] - y) > 0.5:
            out.append([round(x, 1), round(y, 1)])
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3:
        raise HTTPException(400, "The outline is too small. Click points around the whole parcel.")
    if self_intersects(out):  # checked before the area, because a bow-tie has almost no signed area
        raise HTTPException(400, "The outline crosses itself. Undo the last points and click around the parcel in order.")
    if poly_area(out + [out[0]]) < 20:
        raise HTTPException(400, "The outline is too small. Click points around the whole parcel.")
    return out


@app.post("/api/feedback/label")
def post_label(body: LabelIn):
    if body.label not in LABELS:
        raise HTTPException(400, f"label must be one of {LABELS}")
    # What did the system think at the moment this person answered? Saved so we can later
    # measure how often the system's guess was right ("online accuracy").
    g = learner.guess(body.image, body.parcel_id)
    with db() as con:
        ensure_user(con, body.user_id)
        # one current answer per user per parcel: replace earlier answer
        con.execute(
            "DELETE FROM label_feedback WHERE image=? AND parcel_id=? AND user_id=?",
            (body.image, body.parcel_id, body.user_id),
        )
        con.execute(
            "INSERT INTO label_feedback(image,parcel_id,user_id,label,ts,pred_label,pred_conf,model_version,saw_guess)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (body.image, body.parcel_id, body.user_id, body.label, time.time(),
             g["label"] if g else None, g["conf"] if g else None, g["version"] if g else None, int(body.saw_guess)),
        )
        votes = votes_for(con, body.image).get(body.parcel_id, [])
    label, conf, n = consensus(votes)
    learner.maybe_retrain()
    return {"consensus_label": label, "confidence": round(conf, 3), "n_votes": n}


@app.post("/api/feedback/boundary")
def post_boundary(body: BoundaryIn):
    if body.issue not in BOUNDARY_ISSUES:
        raise HTTPException(400, f"issue must be one of {BOUNDARY_ISSUES}")
    with db() as con:
        ensure_user(con, body.user_id)
        con.execute(
            "INSERT INTO boundary_feedback(image,parcel_id,user_id,issue,polygon,ts) VALUES (?,?,?,?,?,?)",
            (body.image, body.parcel_id, body.user_id, body.issue,
             json.dumps(body.polygon) if body.polygon else None, time.time()),
        )
    return {"ok": True}


class RedrawIn(BaseModel):
    image: str
    parcel_id: str
    user_id: str
    polygon: list | None = None  # None = restore the original outline


@app.post("/api/feedback/redraw")
def post_redraw(body: RedrawIn):
    """A person replaces a parcel's outline with a corrected one (or restores the original).
    The latest redraw is the one shown. Every redraw is kept in the database, so the history is not lost."""
    gj = load_geojson(body.image)
    if body.parcel_id not in {f["properties"]["parcel_id"] for f in gj["features"]}:
        raise HTTPException(404, "Unknown parcel")
    poly = None if body.polygon is None else clean_polygon(body.polygon, gj["width"], gj["height"])
    with db() as con:
        ensure_user(con, body.user_id)
        con.execute(
            "INSERT INTO boundary_feedback(image,parcel_id,user_id,issue,polygon,ts) VALUES (?,?,?,?,?,?)",
            (body.image, body.parcel_id, body.user_id, "redrawn", json.dumps(poly) if poly else None, time.time()),
        )
    learner.invalidate(body.image)  # the measurements of this parcel changed, so its guess must be recomputed
    return {"ok": True, "restored": poly is None}


@app.post("/api/feedback/missed")
def post_missed(body: MissedIn):
    if len(body.polygon) < 3:
        raise HTTPException(400, "polygon needs at least 3 points")
    with db() as con:
        ensure_user(con, body.user_id)
        con.execute(
            "INSERT INTO missed_parcels(image,user_id,polygon,label,ts) VALUES (?,?,?,?,?)",
            (body.image, body.user_id, json.dumps(body.polygon), body.label, time.time()),
        )
    if body.label:
        learner.maybe_retrain()
    return {"ok": True}


# ----------------------------------------------------------- the learning part
@app.get("/api/model")
def api_model():
    """Everything the Learning page shows: accuracy, learning curve, history, live agreement."""
    return learner.status()


@app.post("/api/model/retrain")
def api_model_retrain():
    started = learner.retrain_async()
    return {"started": started, **learner.status()}


@app.get("/api/stats")
def stats():
    with db() as con:
        return {
            "label_votes": con.execute("SELECT COUNT(*) FROM label_feedback").fetchone()[0],
            "boundary_reports": con.execute("SELECT COUNT(*) FROM boundary_feedback").fetchone()[0],
            "missed_parcels": con.execute("SELECT COUNT(*) FROM missed_parcels").fetchone()[0],
            "users": con.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        }


@app.get("/api/export")
def export_feedback():
    """All feedback as JSON - this is the training data for the learning loop."""
    with db() as con:
        return {
            "labels": [dict(r) for r in con.execute("SELECT * FROM label_feedback")],
            "boundaries": [dict(r) for r in con.execute("SELECT * FROM boundary_feedback")],
            "missed": [dict(r) for r in con.execute("SELECT * FROM missed_parcels")],
        }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
