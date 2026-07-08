from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.lane_change_test_system.core import (
    DEFAULT_PAYLOAD,
    parse_payload,
    run_lane_change_test,
    sample_points_text,
)


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"


class LaneChangeHandler(BaseHTTPRequestHandler):
    server_version = "LaneChangeTestSystem/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("", "/"):
            self._send_file(STATIC_ROOT / "index.html")
            return
        if parsed.path == "/api/default":
            self._send_json(
                {
                    "payload": DEFAULT_PAYLOAD,
                    "pointsText": sample_points_text(),
                }
            )
            return

        requested = (STATIC_ROOT / parsed.path.lstrip("/")).resolve()
        if STATIC_ROOT in requested.parents and requested.exists() and requested.is_file():
            self._send_file(requested)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            test_input = parse_payload(payload)
            result = run_lane_change_test(test_input)
            self._send_json(result.to_dict())
        except Exception as exc:  # The UI displays the exact validation message.
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="智能车辆变道轨迹自动生成测试系统")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    server = ThreadingHTTPServer((args.host, args.port), LaneChangeHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"测试系统已启动: {url}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n测试系统已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

