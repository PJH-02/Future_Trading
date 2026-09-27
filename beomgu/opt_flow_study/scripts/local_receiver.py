"""브라우저 탭에서 만든 CSV를 로컬 원자료 폴더에 저장하는 1회용 수신기 (127.0.0.1 전용).

앱 내장 브라우저가 파일 다운로드를 막을 때 쓴다. 허용 파일명 하나만 받고, 저장에 성공하면 종료한다.
사용: python scripts/local_receiver.py [--raw-dir PATH] [--port 8765] [--name krx_opt_investor_daily.csv]
브라우저 쪽: fetch('http://127.0.0.1:8765/save', {method: 'POST', body: csvText})
"""
import argparse
import datetime as dt
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optflow.config import resolve_raw_dir  # noqa: E402

ALLOWED_ORIGIN = "https://data.krx.co.kr"


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--name", default="krx_opt_investor_daily.csv")
    ap.add_argument("--max-bytes", type=int, default=50_000_000)
    a = ap.parse_args()
    target = resolve_raw_dir(a.raw_dir) / a.name
    done = threading.Event()

    class H(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self):
            if self.path != "/save":
                self.send_response(404); self._cors(); self.end_headers(); return
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > a.max_bytes:
                self.send_response(413); self._cors(); self.end_headers(); return
            body = self.rfile.read(n)
            text = body.decode("utf-8")
            if not text.startswith("date,investor,cp,"):
                self.send_response(400); self._cors(); self.end_headers(); return
            target.write_bytes(body)                      # 바이트 그대로(줄바꿈 변환 없음)
            with open(target.parent / "_collected_on.txt", "a", encoding="utf-8") as f:
                f.write(f"{dt.date.today()} krx_browser_collect.js -> local_receiver.py {a.name} bytes={n} lines={text.count(chr(10)) + 1}\n")
            self.send_response(200); self._cors()
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"saved": str(target), "bytes": n}).encode())
            print(f"saved {target} ({n} bytes)", flush=True)
            done.set()

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"listening 127.0.0.1:{a.port} → {target}", flush=True)
    done.wait(timeout=1800)
    srv.shutdown()
    print("receiver closed", flush=True)


if __name__ == "__main__":
    main()
