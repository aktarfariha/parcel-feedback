"""Stand-in satellite tile server for testing (NOT real imagery).
Every pixel encodes its own global pixel position so stitching can be verified exactly:
    R = gx % 256,  G = gy % 256,  B = ((gx // 256) * 7 + (gy // 256) * 13) % 256
Serves /{z}/{y}/{x} as PNG on the given port.  Run:  python tests/mock_tiles.py 9000
"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

T = 256


def make_tile(z, x, y):
    gx = x * T + np.arange(T)[None, :].repeat(T, 0)
    gy = y * T + np.arange(T)[:, None].repeat(T, 1)
    r = (gx % 256).astype(np.uint8)
    g = (gy % 256).astype(np.uint8)
    b = (((gx // 256) * 7 + (gy // 256) * 13) % 256).astype(np.uint8)
    return np.dstack([b, g, r])  # BGR for cv2


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            z, y, x = [int(p) for p in self.path.strip("/").split("?")[0].split("/")]
            ok, png = cv2.imencode(".png", make_tile(z, x, y))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(png.tobytes())
        except Exception:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
