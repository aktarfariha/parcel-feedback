"""Builds colab/parcel_detector_colab.ipynb from detector/detector_service.py.
Run:  python colab/build_notebook.py"""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent.parent
service_src = (root / "detector" / "detector_service.py").read_text()

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Parcel detector (Colab GPU)

This notebook runs your SAM parcel detector as a small server. The web app on your computer
sends it satellite images and gets parcel polygons back.

**How to use**
1. `Runtime` -> `Change runtime type` -> **T4 GPU**.
2. Run the cells below **in order** (or `Runtime` -> `Run all`).
3. The last cell prints a **CONNECTION LINK**. Copy it and paste it into the web app
   (*Add new area* -> *Connect detector*).
4. **Leave the last cell running.** If you stop it, or Colab disconnects, run it again
   and paste the new link (the link changes every time).
"""))

cells.append(nbf.v4.new_code_cell("""# 1) Check that a GPU is active
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NO GPU - go to Runtime > Change runtime type > T4 GPU, then run again")"""))

cells.append(nbf.v4.new_code_cell("""# 2) Install (about 1 minute)
!pip install -q git+https://github.com/facebookresearch/segment-anything.git
!pip install -q fastapi "uvicorn[standard]" shapely opencv-python-headless"""))

cells.append(nbf.v4.new_code_cell("""# 3) Download the SAM model (about 2.5 GB, a few minutes; skipped if already downloaded)
import os
if not os.path.exists("sam_vit_h.pth"):
    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth
print("Model file:", round(os.path.getsize("sam_vit_h.pth") / 1e9, 2), "GB")"""))

cells.append(nbf.v4.new_code_cell("%%writefile detector_service.py\n" + service_src))

cells.append(nbf.v4.new_code_cell('''# 4) Start the detector and create the connection link. LEAVE THIS CELL RUNNING.
import os, re, secrets, subprocess, threading, time

os.environ["DETECTOR_TOKEN"] = secrets.token_urlsafe(12)   # shared secret, so only you can use your GPU
os.environ["DETECTOR_BACKEND"] = "sam"

import uvicorn, detector_service as ds
server = uvicorn.Server(uvicorn.Config(ds.app, host="127.0.0.1", port=8001, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()

print("Loading SAM (about 1 minute)...")
while not ds.state["model_loaded"] and not ds.state["error"]:
    time.sleep(2)
if ds.state["error"]:
    raise RuntimeError("Model failed to load: " + ds.state["error"])
print("Model ready on", ds.state["device"])

# public tunnel so the web app on your computer can reach this notebook (no account needed)
if not os.path.exists("cloudflared"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    !chmod +x cloudflared
proc = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8001", "--no-autoupdate"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in proc.stdout:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        url = m.group(0)
        break
threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()  # keep the pipe drained
if not url:
    raise RuntimeError("Could not create the tunnel. Run this cell again.")

print()
print("=" * 70)
print("CONNECTION LINK (copy everything on the next line):")
print(f"{url}?token={os.environ['DETECTOR_TOKEN']}")
print("=" * 70)
print("Paste it into the web app. Keep this cell running.")
while True:
    time.sleep(30)
    active = [j for j in ds.jobs.values() if j["status"] in ("queued", "running")]
    if active:
        print(time.strftime("%H:%M:%S"), f"{len(active)} job(s) in progress")'''))

nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = root / "colab" / "parcel_detector_colab.ipynb"
nbf.write(nb, out)
print("wrote", out, "cells:", len(cells))
