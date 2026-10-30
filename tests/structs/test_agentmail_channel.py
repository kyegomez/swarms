import json
import pytest
import httpx

from swarms.structs.comms_channel import CommsChannel
from swarms.structs.agentmail_channel import AgentMailChannel


class TestAgentMailChannel:
    def test_inherits_comms_channel(self):
        assert issubclass(AgentMailChannel, CommsChannel)

    def test_missing_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("AGENTMAIL_API_KEY", raising=False)
        with pytest.raises(ValueError) as excinfo:
            AgentMailChannel(
                inbox_id="research@agentmail.to",
                to="user@example.com",
                api_key=None,
            )
        assert "api key" in str(excinfo.value).lower()

    def test_api_key_from_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENTMAIL_API_KEY", "test_env_key")
        channel = AgentMailChannel(
            inbox_id="research@agentmail.to",
            to="user@example.com",
        )
        assert channel.api_key == "test_env_key"

    def test_messages_empty_before_first_post(self):
        channel = AgentMailChannel(
            inbox_id="research@agentmail.to",
            to="user@example.com",
            api_key="key_123",
        )
        assert channel.messages() == []

    def test_first_post_sends_to_messages_send_and_stores_ids(self):
        recorded_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            recorded_requests.append(request)
            if request.url.path == "/v0/inboxes/research@agentmail.to/messages/send":
                assert request.headers.get("authorization") == "Bearer secret_api_key"
                body = json.loads(request.content)
                assert body["to"] == ["user@example.com"]
                assert body["subject"] == "Test Thread"
                assert body["text"] == "First turn content."
                assert body["headers"]["X-Swarms-Agent"] == "Researcher"
                return httpx.Response(
                    200,
                    json={
                        "message_id": "msg_001",
                        "thread_id": "thread_abc",
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        channel = AgentMailChannel(
            inbox_id="research@agentmail.to",
            to="user@example.com",
            subject="Test Thread",
            api_key="secret_api_key",
            transport=transport,
        )

        channel.post("Researcher", "First turn content.")

        assert len(recorded_requests) == 1
        assert channel.thread_id == "thread_abc"
        assert channel.last_message_id == "msg_001"

    def test_second_post_replies_to_last_message(self):
        recorded_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            recorded_requests.append(request)
            if "/messages/send" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "message_id": "msg_001",
                        "thread_id": "thread_abc",
                    },
                )
            elif request.url.path == "/v0/inboxes/research@agentmail.to/messages/msg_001/reply":
                body = json.loads(request.content)
                assert body["text"] == "Second turn response."
                assert body["headers"]["X-Swarms-Agent"] == "Writer"
                return httpx.Response(
                    200,
                    json={
                        "message_id": "msg_002",
                        "thread_id": "thread_abc",
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        channel = AgentMailChannel(
            inbox_id="research@agentmail.to",
            to="user@example.com",
            api_key="secret_api_key",
            transport=transport,
        )

        channel.post("Researcher", "First turn content.")
        channel.post("Writer", "Second turn response.")

        assert len(recorded_requests) == 2
        assert channel.last_message_id == "msg_002"
        assert channel.thread_id == "thread_abc"

    def test_messages_paginates_and_sorts_oldest_first(self):
        page1 = {
            "messages": [
                {
                    "id": "msg_002",
                    "timestamp": "2026-10-30T10:05:00Z",
                    "from": "user@example.com",
                    "extracted_text": "Human feedback reply.",
                    "text": "Full text with quotes...",
                    "headers": {},
                }
            ],
            "next_page_token": "token_page_2",
        }
        page2 = {
            "messages": [
                {
                    "id": "msg_001",
                    "timestamp": "2026-10-30T10:00:00Z",
                    "from": "research@agentmail.to",
                    "extracted_text": "First turn content.",
                    "text": "First turn content.",
                    "headers": {"X-Swarms-Agent": "Researcher"},
                }
            ],
            "next_page_token": None,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v0/inboxes/research@agentmail.to/threads/thread_abc":
                if "page_token=token_page_2" in str(request.url):
                    return httpx.Response(200, json=page2)
                return httpx.Response(200, json=page1)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        channel = AgentMailChannel(
            inbox_id="research@agentmail.to",
            to="user@example.com",
            api_key="secret_api_key",
            thread_id="thread_abc",
            transport=transport,
        )

        messages = channel.messages()

        assert len(messages) == 2
        # Oldest first: msg_001 (10:00), then msg_002 (10:05)
        assert messages[0] == {
            "agent_name": "Researcher",
            "content": "First turn content.",
        }
        # Human reply speaker is From header
        assert messages[1] == {
            "agent_name": "user@example.com",
            "content": "Human feedback reply.",
        }

    def test_clear_deletes_thread_and_forgets_ids(self):
        deleted = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "DELETE" and request.url.path == "/v0/inboxes/research@agentmail.to/threads/thread_abc":
                deleted.append(request.url.path)
                return httpx.Response(204)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        channel = AgentMailChannel(
            inbox_id="research@agentmail.to",
            to="user@example.com",
            api_key="secret_api_key",
            thread_id="thread_abc",
            transport=transport,
        )

        channel.clear()

        assert len(deleted) == 1
        assert channel.thread_id is None
        assert channel.last_message_id is None

    def test_non_2xx_raises_http_status_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal error"})

        transport = httpx.MockTransport(handler)
        channel = AgentMailChannel(
            inbox_id="research@agentmail.to",
            to="user@example.com",
            api_key="secret_api_key",
            transport=transport,
        )

        with pytest.raises(httpx.HTTPStatusError):
            channel.post("Researcher", "Task content")
