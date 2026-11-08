"""Serve docs/ locally with caching disabled, so edits show up on a normal refresh.

Usage:
    python scripts/serve_dashboard.py          # http://localhost:8000
    python scripts/serve_dashboard.py 8080
"""
import functools
import http.server
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(NoCacheHandler, directory=str(DOCS))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"serving {DOCS} at http://localhost:{port} (Ctrl+C to stop)")
        server.serve_forever()


if __name__ == "__main__":
    main()
