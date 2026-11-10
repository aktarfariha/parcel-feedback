"""End-to-end test of the learning part with STAND-IN pictures (not real imagery).

    python tests/test_learning.py <scratch_dir>

It simulates a person who labels parcels (with some mistakes), then checks that the system
trains itself, guesses sensibly on parcels and pictures it never saw, and records its own guesses.
"""
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

scratch = Path(sys.argv[1])
data, model = scratch / "data", scratch / "model"
db_path = scratch / "feedback.db"
for p in (db_path,):
    p.unlink(missing_ok=True)
import shutil
shutil.rmtree(model, ignore_errors=True)

import make_synthetic
data.mkdir(parents=True, exist_ok=True)
for s in range(3):
    make_synthetic.make_one(data, f"fake{s + 1}", seed=s + 1)

# an OLD (v1) database: no prediction columns. The app must upgrade it without losing rows.
con = sqlite3.connect(db_path)
con.executescript("""
CREATE TABLE users (user_id TEXT PRIMARY KEY, weight REAL NOT NULL DEFAULT 1.0);
CREATE TABLE label_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, image TEXT NOT NULL, parcel_id TEXT NOT NULL,
  user_id TEXT NOT NULL, label TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE boundary_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, image TEXT NOT NULL, parcel_id TEXT NOT NULL,
  user_id TEXT NOT NULL, issue TEXT NOT NULL, polygon TEXT, ts REAL NOT NULL);
CREATE TABLE missed_parcels (id INTEGER PRIMARY KEY AUTOINCREMENT, image TEXT NOT NULL, user_id TEXT NOT NULL,
  polygon TEXT NOT NULL, label TEXT, ts REAL NOT NULL);
INSERT INTO users(user_id) VALUES ('old');
INSERT INTO label_feedback(image,parcel_id,user_id,label,ts) VALUES ('fake1','fake1_0','old','crop_field',1.0);
""")
con.commit()
con.close()

os.environ.update(PARCEL_DATA_DIR=str(data), PARCEL_DB=str(db_path), PARCEL_MODEL_DIR=str(model))
from fastapi.testclient import TestClient  # noqa: E402
from app import main  # noqa: E402

c = TestClient(main.app)
truth = {}
for s in range(3):
    truth.update(json.loads((data / f"fake{s + 1}_truth.json").read_text()))

# ---- 1. migration kept the old row and added the columns
cols = {r[1] for r in sqlite3.connect(db_path).execute("PRAGMA table_info(label_feedback)")}
assert {"pred_label", "pred_conf", "model_version", "saw_guess"} <= cols, cols
assert sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM label_feedback").fetchone()[0] == 1
print("1. old database upgraded, old row kept")

# ---- 2. before there are enough labels: no model, page still works
st = c.get("/api/model").json()
assert st["trained"] is False
r = c.get("/api/images/fake1/parcels").json()
assert r["features"][0]["properties"]["predicted_label"] is None
print("2. no model yet, parcels endpoint fine")

# ---- 3. a person labels parcels; the system should retrain by itself
random.seed(1)
LABELS = main.LABELS


def human(pid, noise=0.08):
    t = truth[pid]
    return random.choice([l for l in LABELS if l != t]) if random.random() < noise else t


def label(img, pid, user="alice"):
    r = c.post("/api/feedback/label", json={"image": img, "parcel_id": pid, "user_id": user, "label": human(pid)})
    assert r.status_code == 200, r.text


def wait_idle(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get("/api/model").json()
        if not s["training"]:
            return s
        time.sleep(0.5)
    raise TimeoutError("training did not finish")


pids = {f"fake{i}": [f["properties"]["parcel_id"] for f in json.loads((data / f"fake{i}_parcels.geojson").read_text())["features"]] for i in (1, 2, 3)}
for pid in pids["fake1"][1:61]:
    label("fake1", pid)
s = wait_idle()
assert s["trained"], s
v1 = s["version"]
print(f"3. auto-trained after labelling: version {v1}, {s['n_train']} labels, message: {s['message']}")

for pid in pids["fake2"][:45]:
    label("fake2", pid)
time.sleep(0.5)
s = wait_idle()
assert s["version"] > v1, "should have retrained automatically after more answers"
m = s["metrics"]
print(f"   retrained -> version {s['version']} on {s['n_train']} labels")
print(f"   same-area test accuracy : {m['cv_accuracy']:.0%} (always guessing the commonest label: {m['cv_baseline']:.0%})")
print(f"   NEW-area test accuracy  : {m.get('new_area_accuracy')} (baseline {m.get('new_area_baseline')}, pictures {m.get('new_area_images')})")
print("   learning curve:", [(p['n'], p['acc']) for p in m['curve']])
print("   looks at:", m["top_features"])
print("   per label:", {k: (v['n'], v['recall'], v['precision']) for k, v in m['per_class'].items()})
assert m["cv_accuracy"] > m["cv_baseline"] + 0.1
assert m["curve"][-1]["acc"] > m["curve"][0]["acc"], "accuracy should grow with more labels"

# ---- 4. guesses on a picture nobody labelled (a genuinely new area)
r = c.get("/api/images/fake3/parcels?user_id=alice").json()
ok = tot = 0
confs_ok, confs_bad = [], []
for f in r["features"]:
    p = f["properties"]
    assert p["predicted_label"] in LABELS and 0 < p["predicted_confidence"] <= 1 and p["predicted_top"]
    assert p["predicted_seen"] is False
    good = p["predicted_label"] == truth[p["parcel_id"]]
    ok += good
    tot += 1
    (confs_ok if good else confs_bad).append(p["predicted_confidence"])
print(f"4. brand-new picture fake3: {ok / tot:.0%} of {tot} parcels guessed right "
      f"(mean confidence when right {sum(confs_ok) / max(1, len(confs_ok)):.2f}, when wrong {sum(confs_bad) / max(1, len(confs_bad)):.2f})")
assert ok / tot > 0.55

# labelled parcels show the out-of-bag guess, and my_label comes back
r1 = c.get("/api/images/fake1/parcels?user_id=alice").json()
labelled = [f["properties"] for f in r1["features"] if f["properties"]["my_label"]]
assert labelled and all(p["predicted_seen"] for p in labelled)
print(f"   {len(labelled)} labelled parcels in fake1 carry my_label and an honest (out-of-bag) guess")

# ---- 5. active learning uses the model
n = c.get("/api/images/fake3/next?user_id=alice").json()["parcel_id"]
g = {f["properties"]["parcel_id"]: f["properties"]["predicted_confidence"] for f in r["features"]}
rank = sorted(g.values())
print(f"5. next parcel to ask about has confidence {g[n]:.2f} (median confidence is {rank[len(rank) // 2]:.2f})")
assert g[n] <= rank[len(rank) // 2], "should pick a parcel the system is unsure about"

# ---- 6. each answer recorded what the system guessed at that moment
con = sqlite3.connect(db_path)
n_rec, n_agree = con.execute("SELECT COUNT(*), SUM(label = pred_label) FROM label_feedback WHERE pred_label IS NOT NULL").fetchone()
st = c.get("/api/model").json()
print(f"6. {n_rec} answers recorded a system guess; it matched the person {n_agree}/{n_rec}; status online={st['online']}")
assert n_rec > 20 and st["online"]["n"] == n_rec

# ---- 7. a parcel drawn by a person (with a label) becomes a training example
poly = [[100, 100], [160, 100], [160, 150], [100, 150]]
assert c.post("/api/feedback/missed", json={"image": "fake3", "user_id": "alice", "polygon": poly, "label": "pond"}).status_code == 200
X, y, w, groups, ids, skipped = main.learner._collect()
assert any(i[1].startswith("user_added_") for i in ids)
print(f"7. user-drawn parcel included in training data ({len(y)} samples, skipped pictures: {skipped})")

# ---- 8. manual retrain button
before = c.get("/api/model").json()["version"]
assert c.post("/api/model/retrain").json()["started"]
s = wait_idle()
assert s["version"] == before + 1 and not s["error"], s
print(f"8. manual retrain ok -> version {s['version']}; history rows: {len(s['history'])}")

# ---- 9. the saved model loads again after a restart
from app.learner import Learner  # noqa: E402
L2 = Learner(data, model, main.db, LABELS)
L2.startup()
assert L2.bundle and L2.bundle["version"] == s["version"]
print("9. saved model reloaded after restart")

# ---- 10. broken/missing image must not break the page
(data / "fake2.jpg").rename(data / "fake2.jpg.bak")
r = c.get("/api/images/fake2/parcels")
assert r.status_code == 200 and r.json().get("learner_error"), r.text[:200]
(data / "fake2.jpg.bak").rename(data / "fake2.jpg")
print("10. missing image: page data still returned, learner_error reported")
print("\nALL OK")
