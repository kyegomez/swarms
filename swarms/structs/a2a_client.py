import uuid
from typing import Optional

import httpx

VERSION = "1.0"
CARD_PATH = "/.well-known/agent-card.json"


def _text(item: dict) -> str:
    parts = item.get("parts", [])
    return "\n".join(p["text"] for p in parts if "text" in p)


class A2AAgent:
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        headers = {"A2A-Version": VERSION}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(
            headers=headers, timeout=300.0, transport=transport
        )
        card = self.client.get(url.rstrip("/") + CARD_PATH)
        self.card = card.raise_for_status().json()
        self.agent_name = self.card["name"]
        self.agent_description = self.card.get("description", "")
        urls = [
            i["url"]
            for i in self.card.get("supportedInterfaces", [])
            if i.get("protocolBinding") == "JSONRPC"
            and i.get("protocolVersion") == VERSION
        ]
        if not urls:
            raise ValueError(f"{url}: no JSONRPC {VERSION} interface")
        self.url = urls[0]

    def run(self, task: str) -> str:
        mid = str(uuid.uuid4())
        part = {"text": task}
        msg = dict(role="ROLE_USER", messageId=mid, parts=[part])
        body = dict(jsonrpc="2.0", id=mid, method="SendMessage")
        body["params"] = {"message": msg}
        reply = self.client.post(self.url, json=body)
        reply = reply.raise_for_status().json()
        if "error" in reply:
            raise RuntimeError(f"A2A error {reply['error']}")
        result = reply["result"]
        if "message" in result:
            return _text(result["message"])
        status = result["task"]["status"]
        if status["state"] != "TASK_STATE_COMPLETED":
            detail = _text(status.get("message", {}))
            raise RuntimeError(f"A2A {status['state']}: {detail}")
        artifacts = result["task"].get("artifacts", [])
        return "\n".join(_text(a) for a in artifacts)
