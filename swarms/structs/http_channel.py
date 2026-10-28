"""A CommsChannel kept on another server and reached through HTTP calls."""

from typing import Any, Dict, List, Optional

import httpx

from swarms.structs.comms_channel import CommsChannel


class HttpChannel(CommsChannel):
    """A conversation stored by a remote server, so agents on different machines share it.

    The url names one conversation. The server answers three calls on it:
    GET returns every message as a JSON list of {"agent_name": ..., "content": ...},
    oldest first; POST with that JSON object appends one message; DELETE
    removes them all. A non-2xx reply raises httpx.HTTPStatusError.

    Args:
        url: The conversation's endpoint, such as https://host/channels/research.
        headers: Sent with every call, for example an Authorization header.
        timeout: Seconds to wait for each call.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.url = url
        self.client = httpx.Client(
            headers=headers or {}, timeout=timeout
        )

    def post(self, agent_name: str, content: str) -> None:
        """
        Append one message on the server.

        Args:
            agent_name (str): Who is speaking.
            content (str): The message text.
        """
        self.client.post(
            self.url,
            json={"agent_name": agent_name, "content": str(content)},
        ).raise_for_status()

    def messages(self) -> List[Dict[str, Any]]:
        """
        Fetch every message from the server, oldest first.

        Returns:
            List[Dict[str, Any]]: Rows shaped like {"agent_name": ..., "content": ...}.
        """
        response = self.client.get(self.url)
        response.raise_for_status()
        return [
            {
                "agent_name": str(row["agent_name"]),
                "content": str(row["content"]),
            }
            for row in response.json()
        ]

    def clear(self) -> None:
        """
        Delete every message on the server.
        """
        self.client.delete(self.url).raise_for_status()

    def close(self) -> None:
        """
        Close the HTTP connection pool. The server keeps every message.
        """
        self.client.close()
