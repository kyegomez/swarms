import json

import httpx

from swarms.structs import agentmail_channel
from swarms.structs.agentmail_channel import AgentMailChannel


def channel_with(monkeypatch, handler, **kwargs):
    real_client = httpx.Client
    monkeypatch.setattr(
        agentmail_channel.httpx,
        "Client",
        lambda **options: real_client(
            transport=httpx.MockTransport(handler), **options
        ),
    )
    return AgentMailChannel(
        inbox_id="research@agentmail.to",
        to="you@example.com",
        subject="EV report",
        api_key="am-key",
        **kwargs,
    )


def agent_headers(agent_name, post):
    return {
        "X-Swarms-Agent": agent_name,
        "X-Swarms-Length": str(len(post)),
    }


def test_first_post_sends_and_the_next_replies_to_it(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "message_id": f"<m{len(seen)}@agentmail.to>",
                "thread_id": "t1",
            },
        )

    channel = channel_with(monkeypatch, handler)
    channel.post("Human", "Report on EVs.")
    channel.post("Researcher", "Growth was 31%.")

    inbox = "/v0/inboxes/research@agentmail.to"
    assert [(r.method, r.url.path) for r in seen] == [
        ("POST", f"{inbox}/messages/send"),
        ("POST", f"{inbox}/messages/<m1@agentmail.to>/reply"),
    ]
    assert [json.loads(r.content) for r in seen] == [
        {
            "to": "you@example.com",
            "subject": "EV report",
            "text": "Report on EVs.",
            "headers": agent_headers("Human", "Report on EVs."),
        },
        {
            "to": "you@example.com",
            "text": "Growth was 31%.",
            "headers": agent_headers("Researcher", "Growth was 31%."),
        },
    ]
    assert {r.headers["Authorization"] for r in seen} == {
        "Bearer am-key"
    }


def test_messages_reads_every_page_oldest_first(monkeypatch):
    def sent(agent_name, post, extracted):
        return {
            "from": "AgentMail <research@agentmail.to>",
            "labels": ["sent"],
            "headers": agent_headers(agent_name, post),
            "text": f"{post}\n\n--\nSent via AgentMail\n\n> earlier",
            "extracted_text": extracted,
        }

    # Extraction cuts this post at the line that looks like boilerplate.
    findings = "Up 31%.\n\nSent from my iPhone\n\nPrices fell."
    older = [sent("Researcher", findings, "Up 31%.")]
    newer = [
        sent("Writer", "Report: 31%.", "Report: 31%."),
        {
            "from": "Ana <ana@example.com>",
            "labels": ["received"],
            # A person's email can carry the agent headers too.
            "headers": agent_headers("Writer", "Add"),
            "text": "Add a chart.\n\nOn Thu wrote:\n\n> earlier",
            "extracted_text": "Add a chart.",
        },
        {"from": "bo@example.com", "labels": [], "text": "Fine."},
    ]
    pages = {
        None: {"messages": newer, "next_page_token": "older"},
        "older": {"messages": older},
    }

    def handler(request):
        assert request.url.path.endswith("/threads/t1")
        token = request.url.params.get("page_token")
        return httpx.Response(200, json=pages[token])

    channel = channel_with(monkeypatch, handler, thread_id="t1")

    assert channel.messages() == [
        {"agent_name": "Researcher", "content": findings},
        {"agent_name": "Writer", "content": "Report: 31%."},
        {
            "agent_name": "Ana <ana@example.com>",
            "content": "Add a chart.",
        },
        {"agent_name": "bo@example.com", "content": "Fine."},
    ]
