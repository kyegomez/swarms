import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
import pytest

from swarms.structs.agent import Agent
from swarms.structs.http_channel import HttpChannel


@pytest.fixture
def server():
    conversations = {}
    seen_headers = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, body=None):
            data = (
                json.dumps(body).encode() if body is not None else b""
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            seen_headers.append(dict(self.headers))
            if self.path == "/broken":
                self._reply(500, {"error": "down"})
                return
            self._reply(200, conversations.get(self.path, []))

        def do_POST(self):
            seen_headers.append(dict(self.headers))
            row = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            conversations.setdefault(self.path, []).append(row)
            self._reply(201, row)

        def do_DELETE(self):
            conversations.pop(self.path, None)
            self._reply(204)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    httpd.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.seen_headers = seen_headers
    yield httpd
    httpd.shutdown()
    httpd.server_close()


class TestHttpChannel:
    def test_round_trip_keeps_order_and_roles(self, server):
        channel = HttpChannel(f"{server.base}/channels/q3")
        channel.post("user", "Report on EVs.")
        channel.post("Market Analyst", "Growth was 31%.")

        assert channel.messages() == [
            {"agent_name": "user", "content": "Report on EVs."},
            {
                "agent_name": "Market Analyst",
                "content": "Growth was 31%.",
            },
        ]

    def test_a_new_conversation_is_empty(self, server):
        assert (
            HttpChannel(f"{server.base}/channels/new").messages()
            == []
        )

    def test_conversations_at_different_urls_are_kept_apart(
        self, server
    ):
        q3 = HttpChannel(f"{server.base}/channels/q3")
        q4 = HttpChannel(f"{server.base}/channels/q4")
        q3.post("user", "third quarter")
        q4.post("user", "fourth quarter")

        assert [m["content"] for m in q3.messages()] == [
            "third quarter"
        ]
        assert [m["content"] for m in q4.messages()] == [
            "fourth quarter"
        ]

    def test_headers_are_sent_with_every_call(self, server):
        channel = HttpChannel(
            f"{server.base}/channels/q3",
            headers={"Authorization": "Bearer secret"},
        )
        channel.post("user", "hi")
        channel.messages()

        assert [h["Authorization"] for h in server.seen_headers] == [
            "Bearer secret",
            "Bearer secret",
        ]

    def test_clear_empties_the_conversation(self, server):
        channel = HttpChannel(f"{server.base}/channels/q3")
        channel.post("user", "hi")
        channel.clear()
        assert channel.messages() == []

    def test_a_server_error_raises(self, server):
        with pytest.raises(httpx.HTTPStatusError):
            HttpChannel(f"{server.base}/broken").messages()

    def test_agents_on_separate_clients_share_the_conversation(
        self, server
    ):
        url = f"{server.base}/channels/report"

        def agent(name, reply):
            with patch("swarms.agents.tool_manager.LiteLLM"):
                built = Agent(
                    agent_name=name,
                    max_loops=1,
                    print_on=False,
                    verbose=False,
                    persistent_memory=False,
                    channel=HttpChannel(url),
                )
            built.requests = []

            def call_llm(task=None, *args, **kwargs):
                built.requests.append(kwargs.get("messages"))
                return reply

            built.call_llm = call_llm
            return built

        researcher = agent("Researcher", "Growth was 31%.")
        writer = agent("Writer", "Report: 31% growth.")

        researcher.run("Write the report.")
        writer.run("Write the report.")

        assert [
            m["agent_name"] for m in HttpChannel(url).messages()
        ] == [researcher.user_name, "Researcher", "Writer"]
        sent = [m["content"] for m in writer.requests[0]]
        assert "Growth was 31%." in sent
