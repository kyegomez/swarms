import uuid
from typing import Dict, Optional

import httpx

VERSION = "1.0"
BINDING = "JSONRPC"
CARD_PATH = "/.well-known/agent-card.json"
SEND_MESSAGE = "SendMessage"
TASK_COMPLETED = "TASK_STATE_COMPLETED"
TIMEOUT = 300.0


def text_of(item: dict) -> str:
    """Join the text parts of an A2A message or artifact.

    Args:
        item (dict): A message or artifact with a parts list.

    Returns:
        str: The text parts, one per line.
    """
    parts = item.get("parts", [])
    return "\n".join(p["text"] for p in parts if "text" in p)


class A2AClient:
    """A remote A2A agent that runs a task like a local agent.

    Attributes:
        agent_name (str): Name from the remote agent card.
        agent_description (str): Description from the card, or an empty string.
        url (str): JSON-RPC endpoint from the card.
        card (dict): The agent card as the server returned it.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        timeout: float = TIMEOUT,
        headers: Optional[Dict[str, str]] = None,
        card_path: str = CARD_PATH,
        version: str = VERSION,
        agent_name: Optional[str] = None,
        agent_description: Optional[str] = None,
    ):
        """Read the remote agent card and find its JSON-RPC endpoint.

        Args:
            url (str): Base URL of the remote agent.
            api_key (Optional[str]): Bearer token sent on every request.
            transport (Optional[httpx.BaseTransport]): Transport for the HTTP client, used by tests.
            timeout (float): Seconds to wait for each request.
            headers (Optional[Dict[str, str]]): Extra headers sent on every request.
            card_path (str): Path of the agent card under the base URL.
            version (str): A2A protocol version to request and to match on the card.
            agent_name (Optional[str]): Name to use instead of the one on the card.
            agent_description (Optional[str]): Description to use instead of the one on the card.
        """
        self.version = version
        self.client = httpx.Client(
            headers={**self._headers(api_key), **(headers or {})},
            timeout=timeout,
            transport=transport,
        )

        self.card = self._fetch_card(url, card_path)
        self.agent_name = agent_name or self.card["name"]
        self.agent_description = agent_description or self.card.get(
            "description", ""
        )
        self.url = self._rpc_url(url)

    def run(self, task: str) -> str:
        """Send a task to the remote agent and return its reply.

        Args:
            task (str): The user message for the remote agent.

        Returns:
            str: The reply message, or the artifacts of a completed task.
        """
        response = self.client.post(
            self.url, json=self._request(task)
        )
        reply = response.raise_for_status().json()

        if "error" in reply:
            raise RuntimeError(f"A2A error {reply['error']}")
        return self._reply_text(reply["result"])

    def _headers(self, api_key: Optional[str]) -> dict:
        """Build the headers sent on every request.

        Args:
            api_key (Optional[str]): Bearer token, or None for anonymous calls.

        Returns:
            dict: The protocol version header, plus the token when given.
        """
        headers = {"A2A-Version": self.version}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _fetch_card(self, url: str, card_path: str) -> dict:
        """Fetch the remote agent card.

        Args:
            url (str): Base URL of the remote agent.
            card_path (str): Path of the agent card under the base URL.

        Returns:
            dict: The agent card.
        """
        response = self.client.get(url.rstrip("/") + card_path)
        return response.raise_for_status().json()

    def _rpc_url(self, url: str) -> str:
        """Pick the JSON-RPC endpoint the card advertises for this version.

        Args:
            url (str): Base URL of the remote agent, named in the error.

        Returns:
            str: The first matching interface URL.
        """
        for interface in self.card.get("supportedInterfaces", []):
            if (
                interface.get("protocolBinding") == BINDING
                and interface.get("protocolVersion") == self.version
            ):
                return interface["url"]
        raise ValueError(
            f"{url}: no {BINDING} {self.version} interface"
        )

    @staticmethod
    def _request(task: str) -> dict:
        """Build the SendMessage request for a task.

        Args:
            task (str): The user message for the remote agent.

        Returns:
            dict: The JSON-RPC request body.
        """
        message_id = str(uuid.uuid4())
        message = {
            "role": "ROLE_USER",
            "messageId": message_id,
            "parts": [{"text": task}],
        }
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": SEND_MESSAGE,
            "params": {"message": message},
        }

    @staticmethod
    def _reply_text(result: dict) -> str:
        """Read the reply text from a SendMessage result.

        Args:
            result (dict): The result, holding a message or a task.

        Returns:
            str: The message text, or the task's artifacts one per line.
        """
        if "message" in result:
            return text_of(result["message"])

        task = result["task"]
        status = task["status"]
        if status["state"] != TASK_COMPLETED:
            detail = text_of(status.get("message", {}))
            raise RuntimeError(f"A2A {status['state']}: {detail}")

        artifacts = task.get("artifacts", [])
        return "\n".join(text_of(a) for a in artifacts)
