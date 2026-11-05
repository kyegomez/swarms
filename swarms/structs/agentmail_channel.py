"""A CommsChannel kept as an AgentMail or Resend email thread that people can reply to."""

import os
import uuid
from datetime import datetime, timezone
from email.utils import formataddr, parseaddr
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import quote

import httpx

from swarms.structs.comms_channel import CommsChannel
from swarms.structs.smtp_channel import strip_quoted

_HOSTS = {
    "agentmail": "https://api.agentmail.to",
    "resend": "https://api.resend.com",
}


def _segment(value: str) -> str:
    # Message ids can hold "/", "?" or "#", which would end the path segment.
    return quote(value, safe="@")


class AgentMailChannel(CommsChannel):
    """A conversation kept as one email thread in an AgentMail inbox.

    Each post is an email from the inbox to the people in `to`: the first
    starts the thread and the rest reply in it. AgentMail adds its own
    footer and the quoted thread to the body, so two headers let
    messages() read a post back exactly: X-Swarms-Agent names the speaker
    and X-Swarms-Length gives the post's length. People on the thread can
    reply from their own mail client, and messages() returns their reply
    under its From address, without the quoted history. A non-2xx reply
    raises httpx.HTTPStatusError.

    With backend="resend" the thread runs on Resend instead, with a
    full-access API key. Each post is an email from inbox_id, whose domain
    must be verified in Resend for sending and receiving, with the speaker
    as the From name. Resend has no thread object, so messages() returns
    this instance's posts as they were sent, plus the received emails
    whose References name the thread. A thread joined with thread_id
    shows people's replies and this instance's posts only.

    Args:
        inbox_id: The inbox the agents send from, such as
            research@agentmail.to. On Resend, the address to send from.
        to: The address or addresses that get each post and can reply.
        subject: The subject of the first email.
        thread_id: An existing thread to join instead of starting one. On
            Resend, the id every post carries in its References header.
        api_key: The backend's API key. Defaults to the AGENTMAIL_API_KEY
            or RESEND_API_KEY environment variable.
        base_url: The API host. Defaults to https://api.agentmail.to (use
            https://api.agentmail.eu for the EU) or https://api.resend.com.
        timeout: Seconds to wait for each call.
        backend: "agentmail" or "resend".

    Raises:
        ValueError: If the backend is unknown, or no API key is given or set.
    """

    AGENT_HEADER = "X-Swarms-Agent"
    LENGTH_HEADER = "X-Swarms-Length"

    def __init__(
        self,
        inbox_id: str,
        to: Union[str, List[str]],
        subject: str = "Swarms conversation",
        thread_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        backend: Literal["agentmail", "resend"] = "agentmail",
    ):
        if backend not in _HOSTS:
            raise ValueError(
                f"backend must be one of {list(_HOSTS)}, not {backend!r}"
            )
        variable = f"{backend.upper()}_API_KEY"
        api_key = api_key or os.environ.get(variable)
        if not api_key:
            raise ValueError(
                "AgentMailChannel needs an API key: pass api_key or "
                f"set {variable}"
            )
        self.backend = backend
        self.inbox_id = inbox_id
        self.to = to
        self.subject = subject
        self.thread_id = thread_id
        self._reply_to: Optional[str] = None
        self._posted: Dict[str, tuple] = {}
        self._received: Dict[str, Dict[str, Any]] = {}
        base_url = (base_url or _HOSTS[backend]).rstrip("/")
        if backend == "agentmail":
            base_url += f"/v0/inboxes/{_segment(inbox_id)}"
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def post(self, agent_name: str, content: str) -> None:
        """
        Email one message to the thread, starting it on the first post.

        Args:
            agent_name (str): Who is speaking.
            content (str): The message text, sent as the email body.
        """
        text = str(content)
        if self.backend == "resend":
            return self._send(agent_name, text)
        body = {
            "to": self.to,
            "text": text,
            "headers": {
                self.AGENT_HEADER: agent_name,
                self.LENGTH_HEADER: str(len(text)),
            },
        }
        if self.thread_id is None:
            path = "/messages/send"
            body["subject"] = self.subject
        else:
            # A joined thread has no post of ours yet: answer its newest.
            reply_to = (
                self._reply_to or self._thread()["last_message_id"]
            )
            path = f"/messages/{_segment(reply_to)}/reply"
        response = self.client.post(path, json=body)
        response.raise_for_status()
        sent = response.json()
        self.thread_id = sent["thread_id"]
        self._reply_to = sent["message_id"]

    def messages(self) -> List[Dict[str, Any]]:
        """
        Read the whole thread, oldest first. Empty before the first post.

        Returns:
            List[Dict[str, Any]]: Rows shaped like {"agent_name": ..., "content": ...}.
        """
        if self.thread_id is None:
            return []
        if self.backend == "resend":
            return self._read()
        pages = [self._thread()]
        while pages[-1].get("next_page_token"):
            pages.append(self._thread(pages[-1]["next_page_token"]))
        # The first page holds the newest messages, each page oldest first.
        return [
            self._row(message)
            for page in reversed(pages)
            for message in page["messages"]
        ]

    def clear(self) -> None:
        """
        Delete the thread from the inbox, so the next post starts a new
        one. Emails already delivered stay in the recipients' mailboxes.
        Resend cannot delete sent mail, so there the thread is only
        forgotten here.
        """
        if self.thread_id is not None and self.backend == "agentmail":
            self.client.delete(
                f"/threads/{_segment(self.thread_id)}"
            ).raise_for_status()
        self.thread_id = None
        self._reply_to = None
        self._posted = {}

    def close(self) -> None:
        """
        Close the HTTP connection pool. The thread stays in the inbox.
        """
        self.client.close()

    def _thread(self, token: Optional[str] = None) -> Dict[str, Any]:
        response = self.client.get(
            f"/threads/{_segment(self.thread_id)}",
            params={"page_token": token} if token else None,
        )
        response.raise_for_status()
        return response.json()

    def _row(self, message: Dict[str, Any]) -> Dict[str, Any]:
        agent_name = message["from"]
        text = message.get("text") or ""
        # extracted_text leaves out the history a person's reply quotes.
        content = message.get("extracted_text") or text
        # Anyone's email can carry these headers, so only sent mail is believed.
        if "sent" in message.get("labels", []):
            headers = message.get("headers") or {}
            agent_name = headers.get(self.AGENT_HEADER, agent_name)
            length = headers.get(self.LENGTH_HEADER, "")
            if length.isdecimal() and int(length) <= len(text):
                # The post, then AgentMail's footer and the thread it quotes.
                content = text[: int(length)]
        return {"agent_name": agent_name, "content": content}

    def _send(self, agent_name: str, text: str) -> None:
        domain = self.inbox_id.rpartition("@")[2]
        # Resend replaces Message-ID, but replies copy References.
        root = (
            self.thread_id or f"<swarms-{uuid.uuid4().hex}@{domain}>"
        )
        key = uuid.uuid4().hex
        self.client.post(
            "/emails",
            json={
                "from": formataddr((agent_name, self.inbox_id)),
                "to": self.to,
                "subject": self.subject,
                "text": text,
                "headers": {
                    self.AGENT_HEADER: agent_name,
                    "References": root,
                },
            },
            # Resend sends a retried request with the same key only once.
            headers={"Idempotency-Key": key},
        ).raise_for_status()
        self.thread_id = root
        self._posted[key] = (
            datetime.now(timezone.utc),
            agent_name,
            text,
        )

    def _read(self) -> List[Dict[str, Any]]:
        inbox, params = self.inbox_id.lower(), {"limit": 100}
        while True:
            page = self._get("/emails/receiving", params)
            for email in page["data"]:
                to = {parseaddr(a)[1].lower() for a in email["to"]}
                sender = parseaddr(email.get("from", ""))[1].lower()
                # Only someone else's mail to the inbox can answer the thread.
                new = email["id"] not in self._received
                if new and inbox in to and sender != inbox:
                    path = f"/emails/receiving/{email['id']}"
                    self._received[email["id"]] = self._get(path)
            if not page.get("has_more"):
                break
            params["after"] = page["data"][-1]["id"]
        rows = {}
        for email_id, email in self._received.items():
            references = (email.get("headers") or {}).get(
                "references"
            )
            if self.thread_id not in str(references):
                continue
            # Anyone's email can carry X-Swarms-Agent, so it is never read.
            date = datetime.fromisoformat(
                email["created_at"].replace("Z", "+00:00")
            )
            text = strip_quoted(email.get("text") or "")
            rows[email_id] = (date, email["from"], text)
        rows.update(self._posted)
        return [
            {"agent_name": agent_name, "content": content}
            for _, agent_name, content in sorted(rows.values())
        ]

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        response = self.client.get(path, params=params)
        response.raise_for_status()
        return response.json()
