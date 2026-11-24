"""A minimal server an HttpChannel can talk to, keeping each conversation in memory by path."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

conversations = {}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, body=None):
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(200, conversations.get(self.path, []))

    def do_POST(self):
        row = json.loads(
            self.rfile.read(int(self.headers["Content-Length"]))
        )
        conversations.setdefault(self.path, []).append(
            {
                "agent_name": row["agent_name"],
                "content": row["content"],
            }
        )
        self._reply(201, row)

    def do_DELETE(self):
        conversations.pop(self.path, None)
        self._reply(204)


if __name__ == "__main__":
    print("Serving channels on http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
