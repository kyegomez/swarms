"""A CommsChannel backed by an AgentMail email thread over httpx."""

import os
from typing import Any, Dict, List, Optional, Union
import httpx
from loguru import logger

from swarms.structs.comms_channel import CommsChannel


class AgentMailChannel(CommsChannel):
    """An email thread hosted on AgentMail (https://agentmail.to) shared by agents and humans.

    Agents given this channel post their answers as emails in one thread using the
    AgentMail REST API, and read prior turns before each run. Human participants
    on the thread receive posts by email and can reply from their email clients.

    Args:
        inbox_id (str): The AgentMail inbox ID (e.g. "research@agentmail.to").
        to (Union[str, List[str]]): Recipient email address(es).
        subject (str): Email thread subject line (default "Swarms conversation").
        thread_id (Optional[str]): Existing thread ID to join, or None to create on first post.
        api_key (Optional[str]): AgentMail API key (defaults to AGENTMAIL_API_KEY env var).
        base_url (str): AgentMail API base URL (default "https://api.agentmail.to").
        timeout (float): Request timeout in seconds (default 30.0).
        transport (Optional[httpx.BaseTransport]): Optional custom transport (e.g. MockTransport).
    """

    def __init__(
        self,
        inbox_id: str,
        to: Union[str, List[str]],
        subject: str = "Swarms conversation",
        thread_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.agentmail.to",
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.inbox_id = inbox_id
        self.to = [to] if isinstance(to, str) else list(to)
        self.subject = subject
        self.thread_id = thread_id
        self.last_message_id: Optional[str] = None

        resolved_key = api_key or os.environ.get("AGENTMAIL_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key is required for AgentMailChannel. Pass api_key explicitly or set AGENTMAIL_API_KEY env var."
            )
        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
            transport=transport,
        )

    def post(self, agent_name: str, content: str) -> None:
        """Append one message to the AgentMail email thread.

        Args:
            agent_name (str): Who is speaking.
            content (str): The message text.
        """
        if self.thread_id is None:
            url = f"/v0/inboxes/{self.inbox_id}/messages/send"
            payload = {
                "to": self.to,
                "subject": self.subject,
                "text": str(content),
                "headers": {"X-Swarms-Agent": agent_name},
            }
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            self.thread_id = data.get("thread_id")
            self.last_message_id = data.get("message_id") or data.get("id")
            logger.debug(
                f"Created new thread '{self.thread_id}' with message '{self.last_message_id}'"
            )
        else:
            reply_target = self.last_message_id or self.thread_id
            url = f"/v0/inboxes/{self.inbox_id}/messages/{reply_target}/reply"
            payload = {
                "to": self.to,
                "text": str(content),
                "headers": {"X-Swarms-Agent": agent_name},
            }
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            self.last_message_id = data.get("message_id") or data.get("id")
            if not self.thread_id and data.get("thread_id"):
                self.thread_id = data.get("thread_id")
            logger.debug(
                f"Appended reply '{self.last_message_id}' to thread '{self.thread_id}'"
            )

    def messages(self) -> List[Dict[str, Any]]:
        """Fetch every message in this thread from AgentMail, oldest first.

        Returns:
            List[Dict[str, Any]]: Messages shaped like {"agent_name": ..., "content": ...}.
        """
        if self.thread_id is None:
            return []

        url = f"/v0/inboxes/{self.inbox_id}/threads/{self.thread_id}"
        all_raw_messages: List[Dict[str, Any]] = []
        page_token: Optional[str] = None

        while True:
            params = {"page_token": page_token} if page_token else {}
            response = self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                page_messages = data.get("messages", [])
                page_token = data.get("next_page_token")
            elif isinstance(data, list):
                page_messages = data
                page_token = None
            else:
                break

            all_raw_messages.extend(page_messages)
            if not page_token:
                break

        # Process and sort oldest first
        processed = []
        for msg in all_raw_messages:
            content = msg.get("extracted_text") or msg.get("text") or ""
            headers = msg.get("headers") or {}
            agent_name = headers.get("X-Swarms-Agent") or msg.get("from") or ""
            timestamp = msg.get("timestamp") or ""
            processed.append((timestamp, {"agent_name": str(agent_name), "content": str(content)}))

        processed.sort(key=lambda item: item[0])
        return [item[1] for item in processed]

    def clear(self) -> None:
        """Delete the current email thread in AgentMail and reset thread state."""
        if self.thread_id is None:
            return

        url = f"/v0/inboxes/{self.inbox_id}/threads/{self.thread_id}"
        response = self.client.delete(url)
        response.raise_for_status()
        self.thread_id = None
        self.last_message_id = None
        logger.debug("Deleted AgentMail thread and reset channel state.")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
