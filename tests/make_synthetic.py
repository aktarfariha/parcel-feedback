"""Makes STAND-IN satellite pictures with known parcel types, to test the learning code.

These are not real imagery. Classes overlap in colour on purpose (a crop field can look like
tree cover) so accuracy is not trivially 100%.

    python tests/make_synthetic.py <out_dir> [n_pictures]

Writes <out_dir>/<name>.jpg, <out_dir>/<name>_parcels.geojson and <out_dir>/<name>_truth.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# base colour (R, G, B), colour jitter, texture noise, shape
KINDS = {
    "crop_field": ((120, 125, 70), 22, 10, "rect"),
    "pond":       ((55, 75, 70), 14, 4, "rect_small"),
    "house":      ((185, 165, 150), 25, 14, "tiny"),
    "road":       ((150, 140, 120), 14, 6, "strip"),
    "tree_cover": ((50, 90, 50), 18, 24, "blob"),
    "water":      ((60, 85, 110), 12, 3, "big"),
    "other":      ((110, 100, 90), 35, 14, "blob"),
}
MIX = {"crop_field": 0.34, "pond": 0.17, "house": 0.14, "road": 0.08, "tree_cover": 0.15, "water": 0.06, "other": 0.06}


def poly_for(shape, rng, W, H):
    cx, cy = rng.uniform(60, W - 60), rng.uniform(60, H - 60)
    ang = rng.uniform(0, np.pi)
    if shape == "rect":
        w, h = rng.uniform(50, 130), rng.uniform(35, 90)
    elif shape == "rect_small":
        w, h = rng.uniform(25, 60), rng.uniform(20, 50)
    elif shape == "tiny":
        w, h = rng.uniform(8, 18), rng.uniform(8, 16)
    elif shape == "strip":
        w, h = rng.uniform(150, 300), rng.uniform(8, 16)
    elif shape == "big":
        w, h = rng.uniform(120, 220), rng.uniform(70, 140)
    else:  # blob
        w, h = rng.uniform(30, 90), rng.uniform(30, 80)
    if shape == "blob":
        k = rng.randint(7, 11)
        t = np.sort(rng.uniform(0, 2 * np.pi, k))
        r = rng.uniform(0.6, 1.0, k)
        pts = np.stack([np.cos(t) * r * w / 2, np.sin(t) * r * h / 2], 1)
    else:
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        pts = pts + rng.normal(0, 1.2, pts.shape)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = pts @ R.T + [cx, cy]
    return np.clip(pts, 1, [W - 2, H - 2])


def make_one(path, name, seed, n=140, size=1024):
    rng = np.random.RandomState(seed)
    W = H = size
    tint = rng.uniform(0.85, 1.15, 3)  # each picture has its own lighting
    img = np.full((H, W, 3), (95, 105, 70), np.float32) * tint
    img += rng.normal(0, 6, img.shape)
    kinds = list(MIX)
    probs = np.array([MIX[k] for k in kinds])
    feats, truth = [], {}
    for i in range(n):
        k = rng.choice(kinds, p=probs / probs.sum())
        base, jit, noise, shape = KINDS[k]
        pts = poly_for(shape, rng, W, H)
        col = (np.array(base) + rng.normal(0, jit, 3)) * tint
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        layer = col + rng.normal(0, noise, (H, W, 3))
        img[mask > 0] = layer[mask > 0]
        pid = f"{name}_{i}"
        ring = [[round(float(x), 1), round(float(y), 1)] for x, y in pts] + [[round(float(pts[0][0]), 1), round(float(pts[0][1]), 1)]]
        area = float(cv2.contourArea(pts.astype(np.float32)))
        feats.append({"type": "Feature", "id": i, "properties": {"parcel_id": pid, "area_px": round(area, 1), "label": None},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
        truth[pid] = str(k)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    cv2.imwrite(str(path / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    (path / f"{name}_parcels.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "image": name, "width": W, "height": H, "features": feats}))
    (path / f"{name}_truth.json").write_text(json.dumps(truth))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    n_pics = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for s in range(n_pics):
        make_one(out, f"fake{s + 1}", seed=s + 1)
    print("wrote", n_pics, "pictures to", out)
