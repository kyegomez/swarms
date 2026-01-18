import threading
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any


class TelemetryCollector:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.metrics = {"conflicts": 0, "tasks_completed": 0, "tasks_failed": 0, "reclaimed": 0}

    def record_event(self, name: str, payload: Dict[str, Any]):
        with self.lock:
            self.events.append({"ts": time.time(), "name": name, "payload": payload})
            if name == "conflict":
                self.metrics["conflicts"] += 1
            if name == "complete":
                self.metrics["tasks_completed"] += 1
            if name == "fail":
                self.metrics["tasks_failed"] += 1
            if name == "reclaimed":
                self.metrics["reclaimed"] += payload.get("count", 0)

    def snapshot(self):
        with self.lock:
            return {"metrics": dict(self.metrics), "recent_events": list(self.events[-200:])}


class _TelemetryHandler(BaseHTTPRequestHandler):
    collector: TelemetryCollector = None

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = self.collector.snapshot()
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


class TelemetryServer:
    def __init__(self, collector: TelemetryCollector, host: str = "127.0.0.1", port: int = 8008):
        self.collector = collector
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    def start(self):
        handler = _TelemetryHandler
        handler.collector = self.collector
        self._server = HTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._thread.join(timeout=1.0)
