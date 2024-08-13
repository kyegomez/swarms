from typing import Any
from fastapi import status
from starlette.types import Send
from sse_starlette.sse import ensure_bytes, EventSourceResponse, ServerSentEvent

class StreamingResponse(EventSourceResponse):
    """`Response` class for streaming server-sent events.

    Follows the
    [EventSource protocol](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events#interfaces)
    """

    def __init__(
        self,
        content: Any = iter(()),
        *args: Any,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method.

        Args:
            content: The content to stream.
        """
        super().__init__(content=content, *args, **kwargs)

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
            async for data in self.body_iterator:
                chunk = ensure_bytes(data, self.sep)
                with open("log.txt", "a") as log_file:
                    log_file.write(f"chunk: {chunk.decode()}\n")
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
        except Exception as e:
            with open("log.txt", "a") as log_file:
                log_file.write(f"body iterator error: {e}\n")
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