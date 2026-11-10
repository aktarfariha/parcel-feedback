"""Parcel detector service.  Runs where the GPU is (Google Colab, or your own GPU PC).

It wraps the SAME pipeline you validated in Colab (SAM automatic masks -> polygons ->
dedup) behind a small job API so the web app can call it:

    GET  /health                  -> model status
    POST /jobs?max_dim=&points_per_side=&crop_n_layers=   body = image bytes (jpg/png)
                                  -> {"job_id": ...}
    GET  /jobs/{job_id}           -> {"status": queued|running|done|error, ...}

Jobs are asynchronous because a single SAM run can take minutes and tunnels drop
long-running requests. Every call needs the header  X-Token  if DETECTOR_TOKEN is set.

Environment variables:
    DETECTOR_BACKEND   sam (default) | fake   (fake = quick superpixels, only for testing)
    DETECTOR_TOKEN     shared secret (optional but recommended)
    SAM_CHECKPOINT     path to the .pth file      (default sam_vit_h.pth)
    SAM_MODEL_TYPE     vit_h | vit_l | vit_b       (default vit_h)
"""
import os
import queue
import threading
import time
import traceback
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request

BACKEND = os.environ.get("DETECTOR_BACKEND", "sam")
TOKEN = os.environ.get("DETECTOR_TOKEN", "")
CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "sam_vit_h.pth")
MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")

app = FastAPI(title="Parcel detector")
jobs: dict = {}
work_q: "queue.Queue[str]" = queue.Queue()
state = {"backend": BACKEND, "model_loaded": BACKEND == "fake", "device": None, "error": None}
_sam = {"model": None}


def check_token(x_token: str):
    if TOKEN and x_token != TOKEN:
        raise HTTPException(401, "Wrong or missing token")


# ------------------------------------------------------------ the Colab pipeline
def mask_to_polygon(mask, img_area, min_area=30, max_area_frac=0.15, epsilon_frac=0.004):
    from shapely.geometry import Polygon

    m = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        return None
    approx = cv2.approxPolyDP(c, epsilon_frac * cv2.arcLength(c, True), True)
    if len(approx) < 3:
        approx = c
    p = Polygon(approx.reshape(-1, 2))
    if not p.is_valid or p.area < min_area or p.area > img_area * max_area_frac:
        return None
    return p


def dedup_smallest_first(polys, overlap=0.6):
    polys = sorted(polys, key=lambda p: p.area)  # smallest first
    kept = []
    for p in polys:
        if not any(p.intersection(k).area > overlap * min(p.area, k.area) for k in kept):
            kept.append(p)
    return kept


def detect_sam(img_bgr, max_dim=1024, points_per_side=32, crop_n_layers=1):
    import torch
    from segment_anything import SamAutomaticMaskGenerator
    from shapely.affinity import scale as shapely_scale

    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    img = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_area = img.shape[0] * img.shape[1]

    gen = SamAutomaticMaskGenerator(
        model=_sam["model"],
        points_per_side=points_per_side,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.88,
        min_mask_region_area=60,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=2,
    )
    torch.cuda.empty_cache()
    masks = gen.generate(img_rgb)

    polys = []
    for m in masks:
        p = mask_to_polygon(m["segmentation"], img_area)
        if p is not None and not p.is_empty:
            if scale < 1.0:
                p = shapely_scale(p, xfact=1 / scale, yfact=1 / scale, origin=(0, 0))
            polys.append(p)
    return dedup_smallest_first(polys), len(masks)


def detect_fake(img_bgr, **_):
    """Quick superpixel stand-in so the whole system can be tested without a GPU."""
    from shapely.geometry import Polygon
    from skimage.segmentation import slic

    seg = slic(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), n_segments=120, compactness=12, start_label=0)
    polys = []
    for lab in np.unique(seg):
        m = (seg == lab).astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if len(c) < 3:
            continue
        a = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(a) < 3:
            continue
        p = Polygon(a)
        if p.is_valid and p.area >= 80:
            polys.append(p)
    return polys, len(polys)


def load_model():
    if BACKEND == "fake":
        state["device"] = "cpu (fake backend)"
        return
    try:
        import torch
        from segment_anything import sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: no GPU found. SAM on CPU is extremely slow. Enable a GPU runtime.")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
        sam.to(device=device)
        _sam["model"] = sam
        state["device"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        state["model_loaded"] = True
        print(f"SAM {MODEL_TYPE} loaded on {state['device']}")
    except Exception as e:  # report through /health instead of dying silently
        state["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


# ------------------------------------------------------------------- the worker
def worker():
    load_model()
    while True:
        job_id = work_q.get()
        job = jobs.get(job_id)
        if job is None:
            continue
        job["status"], job["started"] = "running", time.time()
        try:
            if not state["model_loaded"]:
                raise RuntimeError(state["error"] or "Model is not loaded")
            arr = np.frombuffer(job.pop("image_bytes"), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode the image")
            h, w = img.shape[:2]
            fn = detect_fake if BACKEND == "fake" else detect_sam
            polys, n_masks = fn(img, **job["params"])
            rings = [[[round(float(x), 1), round(float(y), 1)] for x, y in p.exterior.coords] for p in polys]
            job["result"] = {"width": w, "height": h, "polygons": rings, "raw_masks": n_masks,
                             "seconds": round(time.time() - job["started"], 1)}
            job["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        finally:
            job["finished"] = time.time()


threading.Thread(target=worker, daemon=True).start()


def _gc():
    now = time.time()
    for k in [k for k, v in jobs.items() if v.get("finished") and now - v["finished"] > 3600]:
        jobs.pop(k, None)


# -------------------------------------------------------------------------- API
@app.get("/health")
def health(x_token: str = Header(default="")):
    check_token(x_token)
    return {**state, "queue": work_q.qsize()}


@app.post("/jobs")
async def submit(request: Request, max_dim: int = 1024, points_per_side: int = 32,
                 crop_n_layers: int = 1, x_token: str = Header(default="")):
    check_token(x_token)
    if not state["model_loaded"]:
        raise HTTPException(503, state["error"] or "Model is still loading, try again in a minute")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty body: send the image bytes")
    _gc()
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "created": time.time(), "image_bytes": body,
                    "params": {"max_dim": max(256, min(max_dim, 2048)),
                               "points_per_side": max(8, min(points_per_side, 96)),
                               "crop_n_layers": max(0, min(crop_n_layers, 3))}}
    work_q.put(job_id)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_token: str = Header(default="")):
    check_token(x_token)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (the detector may have restarted)")
    out = {"status": job["status"], "elapsed": round(time.time() - job.get("started", job["created"]), 1)}
    if job["status"] == "queued":
        out["queue_position"] = list(work_q.queue).index(job_id) + 1 if job_id in list(work_q.queue) else 0
    if job["status"] == "done":
        out["result"] = job["result"]
    if job["status"] == "error":
        out["error"] = job["error"]
    return out
