""" Customized Langchain StreamingResponse for Server-Side Events (SSE) """
import asyncio
from functools import partial
from typing import Any

from fastapi import status
from langchain.chains.base import Chain
from sse_starlette import ServerSentEvent
from sse_starlette.sse import EventSourceResponse, ensure_bytes
from starlette.types import Send

from swarms.server.utils import StrEnum


class HTTPStatusDetail(StrEnum):
    """ HTTP error descriptions. """
    INTERNAL_SERVER_ERROR = "Internal Server Error"


class StreamingResponse(EventSourceResponse):
    """`Response` class for streaming server-sent events.

    Follows the
    [EventSource protocol]
    (https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events#interfaces)
    """

    def __init__(
        self,
        *args: Any,
        content: Any = iter(()),
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
                print(f"chunk: {chunk.decode()}")
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
        except Exception as e:
            print(f"body iterator error: {e}")
            chunk = ServerSentEvent(
                data=dict(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=HTTPStatusDetail.INTERNAL_SERVER_ERROR,
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



class ChainRunMode(StrEnum):
    """Enum for LangChain run modes."""

    ASYNC = "async"
    SYNC = "sync"


class LangchainStreamingResponse(StreamingResponse):
    """StreamingResponse class for LangChain resources."""

    def __init__(
        self,
        *args: Any,
        chain: Chain,
        config: dict[str, Any],
        run_mode: ChainRunMode = ChainRunMode.ASYNC,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method.

        Args:
            chain: A LangChain instance.
            config: A config dict.
            *args: Positional arguments to pass to the parent constructor.
            **kwargs: Keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)

        self.chain = chain
        self.config = config

        if run_mode not in list(ChainRunMode):
            raise ValueError(
                f"Invalid run mode '{run_mode}'. Must be one of {list(ChainRunMode)}"
            )

        self.run_mode = run_mode

    async def stream_response(self, send: Send) -> None:
        """Stream LangChain outputs.

        If an exception occurs while iterating over the LangChain, an
        internal server error is sent to the client.

        Args:
            send: The ASGI send callable.
        """
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        if "callbacks" in self.config:
            for callback in self.config["callbacks"]:
                if hasattr(callback, "send"):
                    callback.send = send

        try:
            if self.run_mode == ChainRunMode.ASYNC:
                async for outputs in self.chain.astream(input=self.config):
                    if 'answer' in outputs:
                        chunk = ServerSentEvent(
                            data=outputs['answer']
                        )
                        # Send each chunk with the appropriate body type
                        await send(
                            {
                                "type": "http.response.body", 
                                "body": ensure_bytes(chunk, None), 
                                "more_body": True
                            }
                        )

            else:
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    None, partial(self.chain, **self.config)
                )
            if self.background is not None:
                self.background.kwargs.update({"outputs": outputs})
        except Exception as e:
            print(f"chain runtime error: {e}")
            if self.background is not None:
                self.background.kwargs.update({"outputs": {}, "error": e})
            chunk = ServerSentEvent(
                data=dict(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=HTTPStatusDetail.INTERNAL_SERVER_ERROR,
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
