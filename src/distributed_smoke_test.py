from __future__ import annotations

import json
import threading
import time
import urllib.request

from distributed_controller import Handler, STATE
from http.server import ThreadingHTTPServer


def run_smoke() -> None:
    STATE.running = True
    STATE.best_cv_auc = 0.6

    server = ThreadingHTTPServer(("127.0.0.1", 8877), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    time.sleep(0.25)
    status_raw = urllib.request.urlopen("http://127.0.0.1:8877/api/status", timeout=5).read()
    payload = json.loads(status_raw.decode("utf-8"))
    assert "worker_count" in payload

    reg_req = urllib.request.Request(
        "http://127.0.0.1:8877/api/register",
        data=json.dumps({"worker_name": "smoke"}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    reg_raw = urllib.request.urlopen(reg_req, timeout=5).read()
    reg = json.loads(reg_raw.decode("utf-8"))
    assert "worker_id" in reg

    server.shutdown()
    STATE.running = False
    print("distributed_smoke_test: OK")


if __name__ == "__main__":
    run_smoke()
