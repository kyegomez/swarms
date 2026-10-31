"""A CommsChannel kept as an AgentMail email thread that people can reply to."""

import os
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import httpx

from swarms.structs.comms_channel import CommsChannel


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

    Args:
        inbox_id: The inbox the agents send from, such as
            research@agentmail.to.
        to: The address or addresses that get each post and can reply.
        subject: The subject of the first email.
        thread_id: An existing thread to join instead of starting one.
        api_key: An AgentMail API key. Defaults to the AGENTMAIL_API_KEY
            environment variable.
        base_url: The API host; https://api.agentmail.eu for the EU.
        timeout: Seconds to wait for each call.

    Raises:
        ValueError: If no API key is given or set.
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
        base_url: str = "https://api.agentmail.to",
        timeout: float = 30.0,
    ):
        api_key = api_key or os.environ.get("AGENTMAIL_API_KEY")
        if not api_key:
            raise ValueError(
                "AgentMailChannel needs an API key: pass api_key or "
                "set AGENTMAIL_API_KEY"
            )
        self.inbox_id = inbox_id
        self.to = to
        self.subject = subject
        self.thread_id = thread_id
        self._reply_to: Optional[str] = None
        self.client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/v0/inboxes/"
            f"{_segment(inbox_id)}",
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
        """
        if self.thread_id is not None:
            self.client.delete(
                f"/threads/{_segment(self.thread_id)}"
            ).raise_for_status()
        self.thread_id = None
        self._reply_to = None

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
