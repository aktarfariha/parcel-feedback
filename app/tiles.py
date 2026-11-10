"""Satellite tile download, stitching and georeferencing (Web Mercator, 256 px tiles).

Given a lon/lat box we:
  1. choose the highest zoom (<= MAX_ZOOM) whose image fits in MAX_SIDE_PX,
  2. download the tiles that cover the box,
  3. stitch them and crop to the exact box,
  4. remember (zoom, X0, Y0) so pixels can be converted back to lon/lat.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx
import numpy as np

TILE = 256
MAX_LAT = 85.05112878

# Esri World Imagery by default. Override with the TILE_URL environment variable
# (must contain {z}, {x}, {y}). Check your provider's terms of use.
DEFAULT_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
DEFAULT_ATTRIBUTION = (
    "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
)

TILE_URL = os.environ.get("TILE_URL", DEFAULT_TILE_URL)
ATTRIBUTION = os.environ.get("TILE_ATTRIBUTION", DEFAULT_ATTRIBUTION)
MAX_ZOOM = int(os.environ.get("MAX_ZOOM", "18"))
MIN_ZOOM = int(os.environ.get("MIN_ZOOM", "12"))
MAX_SIDE_PX = int(os.environ.get("MAX_SIDE_PX", "1024"))  # same size SAM works at
MIN_SIDE_PX = int(os.environ.get("MIN_SIDE_PX", "128"))


class AreaError(ValueError):
    """The chosen area cannot be processed (message is shown to the user)."""


# ----------------------------------------------------------------- projection
def lonlat_to_global_px(lon: float, lat: float, z: int):
    """Global pixel coordinates (float) at zoom z."""
    n = TILE * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def global_px_to_lonlat(x: float, y: float, z: int):
    n = TILE * (2 ** z)
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def meters_per_px(lat: float, z: int) -> float:
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2 ** z)


# ------------------------------------------------------------------- planning
def validate_bbox(bbox):
    try:
        west, south, east, north = [float(v) for v in bbox]
    except Exception:
        raise AreaError("bbox must be [west, south, east, north]")
    if not (-180 <= west < east <= 180):
        raise AreaError("West/East longitudes are invalid (west must be smaller than east).")
    if not (-MAX_LAT <= south < north <= MAX_LAT):
        raise AreaError("South/North latitudes are invalid (south must be smaller than north).")
    return west, south, east, north


def plan_area(bbox):
    """Decide zoom and output size for a bbox. Raises AreaError with a friendly message."""
    west, south, east, north = validate_bbox(bbox)
    lat_mid = (south + north) / 2
    chosen = None
    for z in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        x0, y0 = lonlat_to_global_px(west, north, z)
        x1, y1 = lonlat_to_global_px(east, south, z)
        X0, Y0 = math.floor(x0), math.floor(y0)
        w, h = math.ceil(x1) - X0, math.ceil(y1) - Y0
        if max(w, h) <= MAX_SIDE_PX:
            chosen = (z, X0, Y0, w, h)
            break
    if chosen is None:
        raise AreaError(
            f"The area is too large. Select a smaller area (it must fit in about "
            f"{MAX_SIDE_PX}px at zoom {MIN_ZOOM} or finer)."
        )
    z, X0, Y0, w, h = chosen
    if max(w, h) < MIN_SIDE_PX:
        raise AreaError("The area is too small. Select a larger area.")
    tx0, ty0 = X0 // TILE, Y0 // TILE
    tx1, ty1 = (X0 + w - 1) // TILE, (Y0 + h - 1) // TILE
    return {
        "bbox": [west, south, east, north],
        "zoom": z,
        "x0": X0,
        "y0": Y0,
        "width": w,
        "height": h,
        "m_per_px": round(meters_per_px(lat_mid, z), 3),
        "tiles": (tx1 - tx0 + 1) * (ty1 - ty0 + 1),
        "tile_range": [tx0, ty0, tx1, ty1],
        "width_m": round(w * meters_per_px(lat_mid, z)),
        "height_m": round(h * meters_per_px(lat_mid, z)),
    }


# ------------------------------------------------------------------- download
def _fetch_tile(client: httpx.Client, z: int, x: int, y: int):
    url = TILE_URL.format(z=z, x=x, y=y)
    last = None
    for attempt in range(3):
        try:
            r = client.get(url)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise AreaError("A downloaded tile was not a valid image.")
                if img.shape[0] != TILE or img.shape[1] != TILE:
                    img = cv2.resize(img, (TILE, TILE))
                return img
            last = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            last = str(e)
    raise AreaError(f"Could not download satellite tile z{z}/{x}/{y} ({last}). Check your internet connection.")


def fetch_mosaic(plan: dict):
    """Download + stitch + crop. Returns (BGR image, warnings list)."""
    z = plan["zoom"]
    tx0, ty0, tx1, ty1 = plan["tile_range"]
    coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    headers = {"User-Agent": "ParcelFeedbackResearch/1.0 (academic prototype)"}
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=8) as ex:
            tiles = list(ex.map(lambda c: _fetch_tile(client, z, c[0], c[1]), coords))

    mosaic = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), np.uint8)
    flat = 0
    for (tx, ty), t in zip(coords, tiles):
        mosaic[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = t
        if float(t.std()) < 2.0:
            flat += 1

    ox, oy = plan["x0"] - tx0 * TILE, plan["y0"] - ty0 * TILE
    img = mosaic[oy:oy + plan["height"], ox:ox + plan["width"]].copy()

    warnings = []
    if flat > len(tiles) / 2:
        warnings.append(
            "Most downloaded tiles look blank. The provider may have no imagery at this zoom "
            f"({z}) for this place. Try lowering MAX_ZOOM or choosing a different area."
        )
    return img, warnings


# ------------------------------------------------------------- georeferencing
def georef_from_plan(plan: dict) -> dict:
    return {"zoom": plan["zoom"], "x0": plan["x0"], "y0": plan["y0"], "crs": "EPSG:4326",
            "bbox": plan["bbox"], "m_per_px": plan["m_per_px"]}


def pixel_to_lonlat(px: float, py: float, georef: dict):
    return global_px_to_lonlat(georef["x0"] + px, georef["y0"] + py, georef["zoom"])
