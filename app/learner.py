"""The learning part: people's answers become training examples for a model that
guesses what each parcel is, and says how sure it is.

How it works, in plain words
  1. FEATURES    For every parcel we measure simple things from the picture: its colour,
                 its texture, its shape, and how it differs from the land around it.
  2. TRAINING    A Random Forest (many small decision trees that vote) learns the link
                 between those measurements and the labels people gave.
  3. GUESSING    For any parcel the forest gives a probability for every label. The biggest
                 one is the system's guess. Its size is the "confidence".
  4. RETRAINING  When new answers arrive the forest is trained again, in the background.
  5. HONEST TEST Accuracy is only measured on parcels the model did not study while training.

No internet service, no tokens and no GPU are needed. Everything runs on this computer.
"""
import json
import threading
import time
import traceback
from collections import Counter
from contextlib import closing
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

FEATURE_VERSION = 1        # bump when the feature code changes: old saved models are then ignored
MIN_TRAIN = 12             # fewest labelled parcels needed to train
MIN_CLASS_EVAL = 3         # a label needs this many examples before accuracy is measured on it
MIN_AREA_EVAL = 8          # an image needs this many labelled parcels to be used as a "new area" test
AUTO_EVERY = 5             # retrain automatically after this many new or changed answers
FEATURE_MAX_SIDE = 1024    # features are measured on a picture of at most this size (same as SAM)

FEATURE_NAMES = [
    # colour inside the parcel
    "mean_red", "mean_green", "mean_blue", "std_red", "std_green", "std_blue",
    "hue_cos", "hue_sin", "saturation", "brightness", "brightness_spread", "greenness",
    # colour compared with the rest of the picture
    "red_vs_picture", "green_vs_picture", "blue_vs_picture",
    # texture
    "edge_strength", "edge_spread", "fine_texture", "gray_spread", "gray_entropy",
    # the land around the parcel
    "ring_red", "ring_green", "ring_blue", "contrast_red", "contrast_green", "contrast_blue",
    "ring_brightness",
    # shape
    "log_area", "log_size", "elongation", "rectangularity", "solidity", "roundness", "corners",
    "touches_edge",
]
FEATURE_GROUPS = {
    "colour": ["mean_red", "mean_green", "mean_blue", "std_red", "std_green", "std_blue", "hue_cos",
               "hue_sin", "saturation", "brightness", "brightness_spread", "greenness"],
    "colour compared with the picture": ["red_vs_picture", "green_vs_picture", "blue_vs_picture"],
    "texture": ["edge_strength", "edge_spread", "fine_texture", "gray_spread", "gray_entropy"],
    "the land around it": ["ring_red", "ring_green", "ring_blue", "contrast_red", "contrast_green",
                           "contrast_blue", "ring_brightness"],
    "shape and size": ["log_area", "log_size", "elongation", "rectangularity", "solidity", "roundness",
                       "corners", "touches_edge"],
}
N_FEAT = len(FEATURE_NAMES)


# ============================================================ 1. FEATURES
def read_image(path):
    """cv2.imread fails on some Windows paths, so decode the bytes ourselves."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    return img


def extract(img_bgr, rings, src_w, src_h):
    """Measure every parcel. rings = list of [[x, y], ...] in the coordinates of a src_w x src_h picture.
    Returns an array (number of parcels, number of features)."""
    h0, w0 = img_bgr.shape[:2]
    s = min(1.0, FEATURE_MAX_SIDE / max(h0, w0))
    img = cv2.resize(img_bgr, (max(1, round(w0 * s)), max(1, round(h0 * s))), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    H, W = img.shape[:2]
    sx, sy = W / float(src_w), H / float(src_h)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    greenness = (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) / (rgb.sum(axis=2) + 1e-3)
    hue = hsv[..., 0] * (2 * np.pi / 180.0)
    hcos, hsin = np.cos(hue), np.sin(hue)
    g_mean = rgb.reshape(-1, 3).mean(0)
    g_std = rgb.reshape(-1, 3).std(0) + 1e-3

    out = np.zeros((len(rings), N_FEAT), np.float32)
    for i, ring in enumerate(rings):
        pts = (np.asarray(ring, np.float32).reshape(-1, 2) * np.array([sx, sy], np.float32))
        if len(pts) < 3:
            continue
        cnt = pts.reshape(-1, 1, 2)
        area = float(abs(cv2.contourArea(cnt)))
        pi = np.round(pts).astype(np.int32)
        x0, y0 = pi.min(0)
        x1, y1 = pi.max(0)
        rw = int(min(20, max(3, 0.2 * np.sqrt(max(area, 1.0)))))
        pad = rw + 2
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(W, x1 + pad + 1), min(H, y1 + pad + 1)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        mask = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
        cv2.fillPoly(mask, [pi - np.array([cx0, cy0], np.int32)], 1)
        inner = mask
        if mask.sum() > 60:  # shave one pixel so the edge does not blur the colours
            er = cv2.erode(mask, np.ones((3, 3), np.uint8))
            if er.sum() >= 0.5 * mask.sum():
                inner = er
        inner = inner.astype(bool)
        if not inner.any():
            continue
        dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rw + 1, 2 * rw + 1)))
        ringm = (dil > 0) & (mask == 0)

        sl = (slice(cy0, cy1), slice(cx0, cx1))
        px = rgb[sl][inner]
        mean_rgb, std_rgb = px.mean(0), px.std(0)
        hs = hsv[sl][inner]
        gr = gray[sl][inner]
        hist, _ = np.histogram(gr, bins=16, range=(0, 255))
        p = hist / max(hist.sum(), 1)
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        if ringm.sum() >= 5:
            r_mean = rgb[sl][ringm].mean(0)
        else:
            r_mean = mean_rgb

        # shape, measured on the polygon itself
        peri = float(cv2.arcLength(cnt, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        (_, _), (rw_, rh_), _ = cv2.minAreaRect(cnt)
        short, long_ = max(min(rw_, rh_), 1e-3), max(rw_, rh_)
        corners = len(cv2.approxPolyDP(cnt, 0.03 * peri, True)) if peri > 0 else 0
        touches = float(x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2)

        out[i] = [
            *mean_rgb, *std_rgb,
            float(hcos[sl][inner].mean()), float(hsin[sl][inner].mean()), hs[:, 1].mean(), hs[:, 2].mean(), hs[:, 2].std(),
            float(greenness[sl][inner].mean()),
            *((mean_rgb - g_mean) / g_std),
            float(grad[sl][inner].mean()), float(grad[sl][inner].std()), float(lap[sl][inner].mean()), float(gr.std()), entropy,
            *r_mean, *(mean_rgb - r_mean), float(r_mean.mean()),
            float(np.log10(area / (W * H) + 1e-7)), float(np.log10(np.sqrt(area) + 1)), min(long_ / short, 50.0),
            min(area / max(rw_ * rh_, 1e-3), 1.5), min(area / max(hull_area, 1e-3), 1.0),
            min(4 * np.pi * area / max(peri * peri, 1e-3), 1.5), float(corners), touches,
        ]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================ 2. TRAINING + HONEST TESTS
def make_rf(n_trees, oob=False, seed=0):
    return RandomForestClassifier(
        n_estimators=n_trees, max_features="sqrt", class_weight="balanced_subsample",
        oob_score=oob, bootstrap=True, random_state=seed, n_jobs=1,
    )


def _majority(y):
    return Counter(y).most_common(1)[0][0]


def cross_validate(X, y, w, n_trees=80, seed=0):
    """Out-of-fold accuracy: every parcel is guessed by a model that never saw it."""
    classes, counts = np.unique(y, return_counts=True)
    keep = classes[counts >= MIN_CLASS_EVAL]
    mask = np.isin(y, keep)
    if len(keep) < 2 or mask.sum() < MIN_TRAIN:
        return None
    Xe, ye, we = X[mask], y[mask], w[mask]
    k = int(min(5, counts[counts >= MIN_CLASS_EVAL].min()))
    pred = np.empty(len(ye), dtype=object)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xe, ye):
        m = make_rf(n_trees, seed=seed).fit(Xe[tr], ye[tr], sample_weight=we[tr])
        pred[te] = m.predict(Xe[te])
    per = {}
    for c in keep:
        tp = int(((pred == c) & (ye == c)).sum())
        n_true, n_pred = int((ye == c).sum()), int((pred == c).sum())
        per[str(c)] = {"n": n_true, "recall": round(tp / n_true, 3) if n_true else None,
                       "precision": round(tp / n_pred, 3) if n_pred else None}
    return {
        "cv_accuracy": round(float((pred == ye).mean()), 3),
        "cv_baseline": round(float((ye == _majority(ye)).mean()), 3),
        "cv_folds": k, "n_evaluated": int(len(ye)),
        "excluded_rare": int((~mask).sum()),
        "per_class": per,
    }


def new_area_test(X, y, w, groups, n_trees=80, seed=0):
    """Leave one picture out: train on the other pictures, test on this one.
    This is the honest test for 'will it work on an area it has never seen?'."""
    groups = np.asarray(groups)
    imgs, cnt = np.unique(groups, return_counts=True)
    test_imgs = [g for g, c in zip(imgs, cnt) if c >= MIN_AREA_EVAL]
    if len(test_imgs) < 2:
        return None
    ok = tot = base = used = 0
    for g in test_imgs:
        te, tr = groups == g, groups != g
        if tr.sum() < MIN_TRAIN:
            continue
        m = make_rf(n_trees, seed=seed).fit(X[tr], y[tr], sample_weight=w[tr])
        ok += int((m.predict(X[te]) == y[te]).sum())
        base += int((y[te] == _majority(y[tr])).sum())
        tot += int(te.sum())
        used += 1
    if used < 2 or tot == 0:
        return None
    return {"new_area_accuracy": round(ok / tot, 3), "new_area_baseline": round(base / tot, 3),
            "new_area_images": used, "new_area_n": tot}


def learning_curve(X, y, w, repeats=6, n_trees=60, seed=0):
    """'If the system had only seen k labelled parcels, how well would it do on parcels it never saw?'"""
    classes, counts = np.unique(y, return_counts=True)
    mask = np.isin(y, classes[counts >= MIN_CLASS_EVAL])
    Xe, ye, we = X[mask], y[mask], w[mask]
    if len(ye) < 20:
        return None
    if len(ye) > 3000:
        keep = np.random.RandomState(seed).choice(len(ye), 3000, replace=False)
        Xe, ye, we = Xe[keep], ye[keep], we[keep]
    rng = np.random.RandomState(seed)
    accs, bases = {}, []
    try:
        splits = list(StratifiedShuffleSplit(n_splits=repeats, test_size=0.3, random_state=seed).split(Xe, ye))
    except ValueError:
        return None
    for tr, te in splits:
        order = rng.permutation(tr)
        sizes, s = [], 8
        while s < len(order):
            sizes.append(s)
            s *= 2
        sizes.append(len(order))
        bases.append(float((ye[te] == _majority(ye[order])).mean()))
        for s in sizes:
            sub = order[:s]
            m = make_rf(n_trees, seed=seed).fit(Xe[sub], ye[sub], sample_weight=we[sub])
            accs.setdefault(s, []).append(float((m.predict(Xe[te]) == ye[te]).mean()))
    return {"curve": [{"n": s, "acc": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)}
                      for s, v in sorted(accs.items())],
            "curve_baseline": round(float(np.mean(bases)), 3)}


def feature_group_importance(model):
    imp = dict(zip(FEATURE_NAMES, model.feature_importances_))
    groups = {g: float(sum(imp[n] for n in names)) for g, names in FEATURE_GROUPS.items()}
    tot = sum(groups.values()) or 1.0
    return sorted(([g, round(v / tot, 3)] for g, v in groups.items()), key=lambda t: -t[1])


# ============================================================ the Learner (used by the web app)
class Learner:
    def __init__(self, data_dir, model_dir, db_factory, labels):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.db = db_factory
        self.labels = list(labels)
        self.lock = threading.RLock()
        self.train_lock = threading.Lock()
        self.bundle = None
        self.state = {"training": False, "message": "", "error": None}
        self._feat = {}
        self._pred = {}

    # ---------------------------------------------------------------- setup
    def init_tables(self):
        with closing(self.db()) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS model_runs (
                       id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, trigger TEXT,
                       n_train INTEGER, cv_accuracy REAL, cv_baseline REAL, new_area_accuracy REAL,
                       new_area_baseline REAL, seconds REAL)"""
            )
            con.commit()

    def startup(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / "model.joblib"
        if path.exists():
            try:
                b = joblib.load(path)
                if b.get("feature_version") == FEATURE_VERSION and b.get("feature_names") == FEATURE_NAMES:
                    self.bundle = b
            except Exception:
                traceback.print_exc()
        self.maybe_retrain()

    # ---------------------------------------------------------------- features
    def _image_path(self, name):
        for ext in ("jpg", "jpeg", "png"):
            p = self.data_dir / f"{name}.{ext}"
            if p.exists():
                return p
        return None

    def overrides(self, name):
        """{parcel_id: ring} for outlines people corrected with the Redraw tool.
        The latest redraw wins; a redraw saved without a polygon restores the original outline."""
        out = {}
        with closing(self.db()) as con:
            for r in con.execute(
                "SELECT parcel_id, polygon FROM boundary_feedback WHERE image=? AND issue='redrawn' ORDER BY id", (name,)
            ):
                if r["polygon"]:
                    out[r["parcel_id"]] = json.loads(r["polygon"])
                else:
                    out.pop(r["parcel_id"], None)
        return out

    def invalidate(self, name):
        """Forget cached guesses for a picture (called when an outline changes)."""
        with self.lock:
            self._pred.pop(name, None)

    def features_for(self, name):
        """(signature, parcel ids, feature array, (width, height)).
        Cached; recomputed when the files change, and only for the parcels whose outline was redrawn."""
        gj_path = self.data_dir / f"{name}_parcels.geojson"
        img_path = self._image_path(name)
        if not gj_path.exists() or img_path is None:
            raise FileNotFoundError(f"Image or parcel file missing for '{name}'")
        sig = (gj_path.stat().st_mtime_ns, img_path.stat().st_mtime_ns)
        ov = self.overrides(name)
        ov_key = {pid: json.dumps(r) for pid, r in ov.items()}
        with self.lock:
            c = self._feat.get(name)
        if c and c["sig"] == sig and c["ov"] == ov_key:
            return c["sig"], c["ids"], c["X"], c["wh"]
        gj = json.loads(gj_path.read_text())
        img = read_image(img_path)
        sw, sh = gj.get("width") or img.shape[1], gj.get("height") or img.shape[0]
        feats = gj["features"]
        ids = [f["properties"]["parcel_id"] for f in feats]
        rings = [ov.get(pid, f["geometry"]["coordinates"][0]) for pid, f in zip(ids, feats)]
        if c and c["sig"] == sig:  # same files, only some outlines changed: measure just those parcels
            X = c["X"].copy()
            changed = [i for i, pid in enumerate(ids) if c["ov"].get(pid) != ov_key.get(pid)]
            if changed:
                X[changed] = extract(img, [rings[i] for i in changed], sw, sh)
        else:
            X = extract(img, rings, sw, sh)
        entry = {"sig": sig, "ids": ids, "X": X, "wh": (sw, sh), "ov": ov_key}
        with self.lock:
            self._feat[name] = entry
        return sig, ids, X, (sw, sh)

    # ---------------------------------------------------------------- training data
    def _collect(self):
        """Turn the feedback database into (features, label, weight, picture, id) rows."""
        with closing(self.db()) as con:
            rows = con.execute(
                """SELECT f.image, f.parcel_id, f.label, COALESCE(u.weight, 1.0) AS w
                   FROM label_feedback f LEFT JOIN users u ON u.user_id = f.user_id"""
            ).fetchall()
            missed = con.execute("SELECT id, image, polygon, label FROM missed_parcels WHERE label IS NOT NULL").fetchall()
        by_image = {}
        for r in rows:
            if r["label"] in self.labels:
                t = by_image.setdefault(r["image"], {"parcels": {}, "missed": []})["parcels"].setdefault(r["parcel_id"], {})
                t[r["label"]] = t.get(r["label"], 0.0) + r["w"]
        for m in missed:
            if m["label"] in self.labels:
                by_image.setdefault(m["image"], {"parcels": {}, "missed": []})["missed"].append(m)

        X, y, w, groups, ids, skipped = [], [], [], [], [], []
        for img, d in by_image.items():
            try:
                _, pids, F, (sw, sh) = self.features_for(img)
            except Exception:
                skipped.append(img)
                continue
            index = {p: i for i, p in enumerate(pids)}
            for pid, tally in d["parcels"].items():
                if pid not in index:
                    continue
                label = max(tally, key=tally.get)
                X.append(F[index[pid]]); y.append(label); w.append(tally[label] / sum(tally.values()))
                groups.append(img); ids.append((img, pid))
            if d["missed"]:
                rings = [json.loads(m["polygon"]) for m in d["missed"]]
                MF = extract(read_image(self._image_path(img)), rings, sw, sh)
                for m, row in zip(d["missed"], MF):
                    X.append(row); y.append(m["label"]); w.append(1.0)
                    groups.append(img); ids.append((img, f"user_added_{m['id']}"))
        return (np.array(X, np.float32).reshape(-1, N_FEAT), np.array(y, dtype=object), np.array(w, np.float32),
                np.array(groups, dtype=object), ids, skipped)

    # ---------------------------------------------------------------- retraining
    def retrain(self, trigger="manual"):
        if not self.train_lock.acquire(blocking=False):
            return False  # already training
        t0 = time.time()
        try:
            self.state.update(training=True, error=None, message="Collecting the labelled parcels...")
            X, y, w, groups, ids, skipped = self._collect()
            n = len(y)
            if n < MIN_TRAIN or len(set(y)) < 2:
                self.state["message"] = (f"Not enough labels yet: it needs at least {MIN_TRAIN} labelled parcels "
                                         f"with 2 or more different labels (it has {n}).")
                return False
            self.state["message"] = "Testing on parcels it has not studied..."
            metrics = {"n_train": n, "n_pictures": int(len(set(groups))),
                       "class_counts": dict(Counter(y)), "skipped_pictures": skipped}
            for part in (cross_validate(X, y, w), new_area_test(X, y, w, groups)):
                if part:
                    metrics.update(part)
            self.state["message"] = "Measuring how accuracy grows with more labels..."
            curve = learning_curve(X, y, w)
            if curve:
                metrics.update(curve)

            self.state["message"] = "Training the final model..."
            model = make_rf(200, oob=True).fit(X, y, sample_weight=w)
            metrics["top_features"] = feature_group_importance(model)
            oob = {}
            dec = getattr(model, "oob_decision_function_", None)
            if dec is not None:
                for key, row in zip(ids, dec):
                    if not np.isnan(row).any():
                        oob[key] = [float(v) for v in row]
            with self.lock:
                version = (self.bundle["version"] + 1) if self.bundle else 1
            bundle = {"version": version, "trained_at": time.time(), "feature_version": FEATURE_VERSION,
                      "feature_names": FEATURE_NAMES, "model": model, "classes": [str(c) for c in model.classes_],
                      "oob": oob, "n_train": n, "metrics": metrics}
            self.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, self.model_dir / "model.joblib")
            with closing(self.db()) as con:
                con.execute(
                    "INSERT INTO model_runs(ts, trigger, n_train, cv_accuracy, cv_baseline, new_area_accuracy,"
                    " new_area_baseline, seconds) VALUES (?,?,?,?,?,?,?,?)",
                    (bundle["trained_at"], trigger, n, metrics.get("cv_accuracy"), metrics.get("cv_baseline"),
                     metrics.get("new_area_accuracy"), metrics.get("new_area_baseline"), round(time.time() - t0, 1)),
                )
                con.commit()
            with self.lock:
                self.bundle = bundle
                self._pred.clear()
            self.state["message"] = f"Trained on {n} labelled parcels in {time.time() - t0:.0f} s."
            return True
        except Exception as e:
            traceback.print_exc()
            self.state["error"] = f"{type(e).__name__}: {e}"
            self.state["message"] = "Training failed."
            return False
        finally:
            self.state["training"] = False
            self.train_lock.release()

    def _count_new(self, since):
        with closing(self.db()) as con:
            a = con.execute("SELECT COUNT(*) FROM label_feedback WHERE ts > ?", (since,)).fetchone()[0]
            b = con.execute("SELECT COUNT(*) FROM missed_parcels WHERE label IS NOT NULL AND ts > ?", (since,)).fetchone()[0]
        return a + b

    def _count_labelled(self):
        with closing(self.db()) as con:
            a = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT image, parcel_id FROM label_feedback)").fetchone()[0]
            b = con.execute("SELECT COUNT(*) FROM missed_parcels WHERE label IS NOT NULL").fetchone()[0]
        return a + b

    def maybe_retrain(self):
        """Called after every new answer. Starts a background retraining when enough has changed."""
        if self.state["training"]:
            return
        b = self.bundle
        go = (self._count_labelled() >= MIN_TRAIN) if b is None else (self._count_new(b["trained_at"]) >= AUTO_EVERY)
        if go:
            threading.Thread(target=self.retrain, args=("auto",), daemon=True).start()

    def retrain_async(self):
        if self.state["training"]:
            return False
        threading.Thread(target=self.retrain, args=("manual",), daemon=True).start()
        self.state.update(training=True, message="Starting...")
        return True

    # ---------------------------------------------------------------- guessing
    def predict_image(self, name):
        """{parcel_id: {label, conf, top, oob, entropy}} or None when no model exists yet.
        For parcels people already labelled the guess comes from the trees that did NOT see that parcel
        (out-of-bag), so it shows what the system would have said without your answer."""
        b = self.bundle
        if b is None:
            return None
        with self.lock:
            c = self._pred.get(name)
            if c and c[0] == b["version"]:
                return c[1]
        _, ids, X, _ = self.features_for(name)
        P = b["model"].predict_proba(X) if len(X) else np.zeros((0, len(b["classes"])))
        classes = b["classes"]
        out = {}
        for i, pid in enumerate(ids):
            o = b["oob"].get((name, pid))
            row = np.asarray(o) if o is not None else P[i]
            order = np.argsort(-row)[:3]
            out[pid] = {
                "label": classes[order[0]], "conf": round(float(row[order[0]]), 3),
                "top": [[classes[j], round(float(row[j]), 3)] for j in order if row[j] > 0.0005],
                "oob": o is not None, "entropy": float(-(row * np.log(row + 1e-12)).sum()),
            }
        with self.lock:
            self._pred[name] = (b["version"], out)
        return out

    def guess(self, name, parcel_id):
        try:
            g = (self.predict_image(name) or {}).get(parcel_id)
        except Exception:
            return None
        if g is None or self.bundle is None:
            return None
        return {**g, "version": self.bundle["version"]}

    # ---------------------------------------------------------------- status for the Learning page
    def status(self):
        b = self.bundle
        s = {"labels": self.labels, "trained": b is not None, "training": self.state["training"],
             "message": self.state["message"], "error": self.state["error"],
             "min_train": MIN_TRAIN, "auto_every": AUTO_EVERY, "n_labelled_now": self._count_labelled()}
        if b:
            s.update(version=b["version"], trained_at=b["trained_at"], n_train=b["n_train"], metrics=b["metrics"],
                     labels_waiting=self._count_new(b["trained_at"]))
        with closing(self.db()) as con:
            s["history"] = [dict(r) for r in con.execute(
                "SELECT id, ts, trigger, n_train, cv_accuracy, cv_baseline, new_area_accuracy FROM model_runs ORDER BY id DESC LIMIT 20")][::-1]
            rows = con.execute(
                "SELECT label, pred_label, saw_guess FROM label_feedback WHERE pred_label IS NOT NULL ORDER BY ts").fetchall()
        agree = [r["label"] == r["pred_label"] for r in rows]
        blind = [r["label"] == r["pred_label"] for r in rows if not r["saw_guess"]]
        s["online"] = {
            "n": len(agree), "agree": round(sum(agree) / len(agree), 3) if agree else None,
            "recent_n": len(agree[-50:]), "recent_agree": round(sum(agree[-50:]) / len(agree[-50:]), 3) if agree else None,
            "blind_n": len(blind), "blind_agree": round(sum(blind) / len(blind), 3) if blind else None,
        }
        return s
