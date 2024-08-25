""" Customized Langchain StreamingResponse for Server-Side Events (SSE) """
import asyncio
from functools import partial
from typing import Any, AsyncIterator

from fastapi import status
from sse_starlette import ServerSentEvent
from sse_starlette.sse import EventSourceResponse, ensure_bytes
from starlette.types import Send



class StreamingResponse(EventSourceResponse):
    """`Response` class for streaming server-sent events.

    Follows the
    [EventSource protocol]
    (https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events#interfaces)
    """

    def __init__(
        self,
        content: AsyncIterator[Any],
    ) -> None:
        """Constructor method.

        Args:
            content: The content to stream.
        """
        super().__init__(content=content)
        self.content = content
        
    async def stream_response(self, send: Send) -> None:
        """Streams data chunks to client by iterating over `content`.

        If an exception occurs while iterating over `content`, an
        internal server error is sent to the client.

        Args:
            send: The send function from the ASGI framework.
        """
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        try:
            async for data in self.content:
                chunk = ensure_bytes(data, self.sep)
                print(f"chunk: {chunk.decode()}")
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
        except Exception as e:
            print(f"body iterator error: {e}")
            chunk = ServerSentEvent(
                data=dict(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal Server Error",
                ),
                event="error",
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": ensure_bytes(chunk, None),
                    "more_body": True,
                }
            )

        await send({"type": "http.response.body", "body": b"", "more_body": False})

    def enable_compression(self, force: bool=False):
        raise NotImplementedError
