"""Serve docs/ locally with caching disabled, so edits show up on a normal refresh.

Listens on both loopback addresses (127.0.0.1 and ::1) so http://localhost:<port>
works whichever one the browser resolves "localhost" to. If the port is already
taken by a running instance, it says so and exits cleanly instead of crashing.

Usage:
    python scripts/serve_dashboard.py          # http://localhost:8000
    python scripts/serve_dashboard.py 8080
"""
import functools
import http.server
import socket
import sys
import threading
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, format, *args):  # keep the terminal quiet
        pass


class IPv6Server(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    url = f"http://localhost:{port}"

    if _port_in_use(port):
        print(f"dashboard already running at {url}", flush=True)
        return

    handler = functools.partial(NoCacheHandler, directory=str(DOCS))
    servers = [http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)]
    try:
        servers.append(IPv6Server(("::1", port), handler))
    except OSError:
        pass  # IPv6 loopback unavailable; IPv4 alone is fine

    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    print(f"serving {DOCS} at {url} (Ctrl+C to stop)", flush=True)
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.server_close()


if __name__ == "__main__":
    main()
