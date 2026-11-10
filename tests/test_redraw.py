"""Test of the Redraw-boundary feature with STAND-IN pictures.

    python tests/test_redraw.py <scratch_dir>
"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

scratch = Path(sys.argv[1])
data, model, db_path = scratch / "data", scratch / "model", scratch / "feedback.db"
db_path.unlink(missing_ok=True)
shutil.rmtree(model, ignore_errors=True)
data.mkdir(parents=True, exist_ok=True)

import make_synthetic

make_synthetic.make_one(data, "fake1", seed=1)
# give the picture a map location so GeoJSON export can be tested
gjp = data / "fake1_parcels.geojson"
g = json.loads(gjp.read_text())
g["georef"] = {"zoom": 18, "x0": 200000, "y0": 120000, "crs": "EPSG:4326", "bbox": [0, 0, 1, 1], "m_per_px": 0.5}
gjp.write_text(json.dumps(g))

os.environ.update(PARCEL_DATA_DIR=str(data), PARCEL_DB=str(db_path), PARCEL_MODEL_DIR=str(model))
from fastapi.testclient import TestClient  # noqa: E402
from app import main  # noqa: E402

c = TestClient(main.app)
truth = json.loads((data / "fake1_truth.json").read_text())


def feats(user=""):
    return {f["properties"]["parcel_id"]: f for f in c.get(f"/api/images/fake1/parcels?user_id={user}").json()["features"]}


def post(poly, pid="fake1_5", user="alice"):
    return c.post("/api/feedback/redraw", json={"image": "fake1", "parcel_id": pid, "user_id": user, "polygon": poly})


before = feats()["fake1_5"]
orig_ring = before["geometry"]["coordinates"][0]
print("original outline has", len(orig_ring) - 1, "points, area", before["properties"]["area_px"])

# 1. a valid redraw replaces the outline everywhere
new = [[100, 100], [300, 100], [300, 220], [100, 220]]
r = post(new)
assert r.status_code == 200, r.text
f = feats()["fake1_5"]
assert f["geometry"]["coordinates"][0] == new + [new[0]], f["geometry"]
assert f["properties"]["edited"] is True and f["properties"]["original_geometry"]["coordinates"][0] == orig_ring
assert abs(f["properties"]["area_px"] - 200 * 120) < 1
print("1. redraw replaces the outline; original kept; area recomputed:", f["properties"]["area_px"])

# 2. other parcels are untouched
others = {k: v for k, v in feats().items() if k != "fake1_5"}
assert not any(v["properties"].get("edited") for v in others.values())
print("2. other parcels untouched")

# 3. invalid outlines are rejected with a friendly message
bad = {
    "two points": [[1, 1], [50, 50]],
    "bow-tie": [[100, 100], [200, 200], [200, 100], [100, 200]],
    "tiny": [[100, 100], [101, 100], [101, 101], [100, 101]],
    "not numbers": [["a", "b"], [1, 2], [3, 4]],
}
for k, poly in bad.items():
    r = post(poly)
    assert r.status_code == 400, (k, r.status_code, r.text)
    print(f"3. rejected ({k}):", r.json()["detail"])
assert post(new, pid="nope").status_code == 404
assert feats()["fake1_5"]["geometry"]["coordinates"][0] == new + [new[0]], "bad redraws must not change anything"

# 4. points outside the picture are clamped inside it
post([[-50, -50], [100, -50], [100, 100], [-50, 100]], pid="fake1_6")
ring = feats()["fake1_6"]["geometry"]["coordinates"][0]
assert min(p[0] for p in ring) >= 0 and min(p[1] for p in ring) >= 0
print("4. out-of-picture points clamped:", ring)

# 5. the latest redraw wins, even from another person
new2 = [[100, 100], [250, 100], [250, 180], [100, 180]]
post(new2, user="bob")
assert feats()["fake1_5"]["geometry"]["coordinates"][0] == new2 + [new2[0]]
print("5. latest redraw wins")

# 6. GeoJSON export uses the corrected outline and area
exp = c.get("/api/images/fake1/export.geojson").json()
e5 = [x for x in exp["features"] if x["properties"]["parcel_id"] == "fake1_5"][0]
assert e5["properties"]["outline_edited"] is True
assert abs(e5["properties"]["area_m2"] - 150 * 80 * 0.25) < 1, e5["properties"]["area_m2"]
print("6. export uses the corrected outline, area_m2 =", e5["properties"]["area_m2"])

# 7. the learner measures the corrected outline (only that parcel changes)
L = main.learner
_, ids, X1, _ = L.features_for("fake1")
i5 = ids.index("fake1_5")
post([[100, 100], [140, 100], [140, 130], [100, 130]])
_, _, X2, _ = L.features_for("fake1")
diff = [ids[i] for i in range(len(ids)) if abs(X1[i] - X2[i]).sum() > 1e-6]
assert diff == ["fake1_5"], diff
print("7. features recomputed for the redrawn parcel only:", diff)

# 8. restoring brings back the original outline and original measurements
r = c.post("/api/feedback/redraw", json={"image": "fake1", "parcel_id": "fake1_5", "user_id": "alice", "polygon": None})
assert r.json()["restored"] is True
f = feats()["fake1_5"]
assert f["geometry"]["coordinates"][0] == orig_ring and not f["properties"].get("edited")
_, _, X3, _ = L.features_for("fake1")
from app import learner as LM  # noqa: E402
direct = LM.extract(LM.read_image(data / "fake1.jpg"), [orig_ring], 1024, 1024)[0]
assert abs(X3[i5] - direct).sum() < 1e-4, "restored features must equal features of the original outline"
others_same = all(abs(X3[i] - X1[i]).sum() < 1e-6 for i in range(len(ids)) if i != i5)
assert others_same
print("8. restore returns the original outline and its original measurements")

# 9. predictions are refreshed after a redraw (model trained from labels)
import random
random.seed(2)
for pid in list(truth)[:40]:
    c.post("/api/feedback/label", json={"image": "fake1", "parcel_id": pid, "user_id": "alice", "label": truth[pid]})
import time
t0 = time.time()
while time.time() - t0 < 60 and (c.get("/api/model").json()["training"] or not c.get("/api/model").json()["trained"]):
    time.sleep(0.5)
assert c.get("/api/model").json()["trained"]
pid = list(truth)[60]
g0 = feats()[pid]["properties"]["predicted_confidence"]
post([[300, 300], [320, 300], [320, 315], [300, 315]], pid=pid)
g1 = feats()[pid]["properties"]["predicted_confidence"]
print(f"9. guess for a redrawn parcel recomputed: confidence {g0} -> {g1}")
assert g0 != g1

# 10. 'not a parcel' reports are visible to the page (it dims those parcels)
c.post("/api/feedback/boundary", json={"image": "fake1", "parcel_id": "fake1_9", "user_id": "alice", "issue": "not_a_parcel"})
assert feats()["fake1_9"]["properties"]["boundary_issues"].get("not_a_parcel") == 1
print("10. not_a_parcel report visible in parcel data")
print("\nALL OK")
