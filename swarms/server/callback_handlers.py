# modified from Lanarky sourcecode https://github.com/auxon/lanarky
from typing import Any, Optional

from fastapi.websockets import WebSocket
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.globals import get_llm_cache
from langchain.schema.document import Document
from pydantic import BaseModel
from starlette.types import Message, Send
from sse_starlette.sse import ensure_bytes, ServerSentEvent
from swarms.server.utils import StrEnum, model_dump_json


class LangchainEvents(StrEnum):
    SOURCE_DOCUMENTS = "source_documents"


class BaseCallbackHandler(AsyncCallbackHandler):
    """Base callback handler for streaming / async applications."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.llm_cache_used = get_llm_cache() is not None

    @property
    def always_verbose(self) -> bool:
        """Verbose mode is always enabled."""
        return True

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Any: ...


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""

    def __init__(
        self,
        *,
        send: Send = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method.

        Args:
            send: The ASGI send callable.
            **kwargs: Keyword arguments to pass to the parent constructor.
        """
        super().__init__(**kwargs)

        self._send = send
        self.streaming = None

    @property
    def send(self) -> Send:
        return self._send

    @send.setter
    def send(self, value: Send) -> None:
        """Setter method for send property."""
        if not callable(value):
            raise ValueError("value must be a Callable")
        self._send = value

    def _construct_message(self, data: str, event: Optional[str] = None) -> Message:
        """Constructs message payload.

        Args:
            data: The data payload.
            event: The event name.
        """
        chunk = ServerSentEvent(data=data, event=event)
        return {
            "type": "http.response.body",
            "body": ensure_bytes(chunk, None),
            "more_body": True,
        }


class TokenStreamMode(StrEnum):
    TEXT = "text"
    JSON = "json"


class TokenEventData(BaseModel):
    """Event data payload for tokens."""

    token: str = ""


def get_token_data(token: str, mode: TokenStreamMode) -> str:
    """Get token data based on mode.

    Args:
        token: The token to use.
        mode: The stream mode.
    """
    if mode not in list(TokenStreamMode):
        raise ValueError(f"Invalid stream mode: {mode}")

    if mode == TokenStreamMode.TEXT:
        return token
    else:
        return model_dump_json(TokenEventData(token=token))


class TokenStreamingCallbackHandler(StreamingCallbackHandler):
    """Callback handler for streaming tokens."""

    def __init__(
        self,
        *,
        output_key: str,
        mode: TokenStreamMode = TokenStreamMode.JSON,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method.

        Args:
            output_key: chain output key.
            mode: The stream mode.
            **kwargs: Keyword arguments to pass to the parent constructor.
        """
        super().__init__(**kwargs)

        self.output_key = output_key

        if mode not in list(TokenStreamMode):
            raise ValueError(f"Invalid stream mode: {mode}")
        self.mode = mode

    async def on_chain_start(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Run when chain starts running."""
        self.streaming = False

    async def on_llm_new_token(self, token: str, **kwargs: dict[str, Any]) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if not self.streaming:
            self.streaming = True

        if self.llm_cache_used:  # cache missed (or was never enabled) if we are here
            self.llm_cache_used = False

        message = self._construct_message(
            data=get_token_data(token, self.mode), event="completion"
        )
        await self.send(message)

    async def on_chain_end(
        self, outputs: dict[str, Any], **kwargs: dict[str, Any]
    ) -> None:
        """Run when chain ends running.

        Final output is streamed only if LLM cache is enabled.
        """
        if self.llm_cache_used or not self.streaming:
            if self.output_key in outputs:
                message = self._construct_message(
                    data=get_token_data(outputs[self.output_key], self.mode),
                    event="completion",
                )
                await self.send(message)
            else:
                raise KeyError(f"missing outputs key: {self.output_key}")


class SourceDocumentsEventData(BaseModel):
    """Event data payload for source documents."""

    source_documents: list[dict[str, Any]]


class SourceDocumentsStreamingCallbackHandler(StreamingCallbackHandler):
    """Callback handler for streaming source documents."""

    async def on_chain_end(
        self, outputs: dict[str, Any], **kwargs: dict[str, Any]
    ) -> None:
        """Run when chain ends running."""
        if "source_documents" in outputs:
            if not isinstance(outputs["source_documents"], list):
                raise ValueError("source_documents must be a list")
            if not isinstance(outputs["source_documents"][0], Document):
                raise ValueError("source_documents must be a list of Document")

            # NOTE: langchain is using pydantic_v1 for `Document`
            source_documents: list[dict] = [
                document.dict() for document in outputs["source_documents"]
            ]
            message = self._construct_message(
                data=model_dump_json(
                    SourceDocumentsEventData(source_documents=source_documents)
                ),
                event=LangchainEvents.SOURCE_DOCUMENTS,
            )
            await self.send(message)


class FinalTokenStreamingCallbackHandler(
    TokenStreamingCallbackHandler, FinalStreamingStdOutCallbackHandler
):
    """Callback handler for streaming final answer tokens.

    Useful for streaming responses from Langchain Agents.
    """

    def __init__(
        self,
        *,
        answer_prefix_tokens: Optional[list[str]] = None,
        strip_tokens: bool = True,
        stream_prefix: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method.

        Args:
            answer_prefix_tokens: The answer prefix tokens to use.
            strip_tokens: Whether to strip tokens.
            stream_prefix: Whether to stream the answer prefix.
            **kwargs: Keyword arguments to pass to the parent constructor.
        """
        super().__init__(output_key=None, **kwargs)

        FinalStreamingStdOutCallbackHandler.__init__(
            self,
            answer_prefix_tokens=answer_prefix_tokens,
            strip_tokens=strip_tokens,
            stream_prefix=stream_prefix,
        )

    async def on_llm_start(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False
        self.streaming = False

    async def on_llm_new_token(self, token: str, **kwargs: dict[str, Any]) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if not self.streaming:
            self.streaming = True

        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                message = self._construct_message(
                    data=get_token_data("".join(self.last_tokens), self.mode),
                    event="completion",
                )
                await self.send(message)

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            message = self._construct_message(
                data=get_token_data(token, self.mode), event="completion"
            )
            await self.send(message)


class WebSocketCallbackHandler(BaseCallbackHandler):
    """Callback handler for websocket sessions."""

    def __init__(
        self,
        *,
        mode: TokenStreamMode = TokenStreamMode.JSON,
        websocket: WebSocket = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method.

        Args:
            mode: The stream mode.
            websocket: The websocket to use.
            **kwargs: Keyword arguments to pass to the parent constructor.
        """
        super().__init__(**kwargs)

        if mode not in list(TokenStreamMode):
            raise ValueError(f"Invalid stream mode: {mode}")
        self.mode = mode

        self._websocket = websocket
        self.streaming = None

    @property
    def websocket(self) -> WebSocket:
        return self._websocket

    @websocket.setter
    def websocket(self, value: WebSocket) -> None:
        """Setter method for send property."""
        if not isinstance(value, WebSocket):
            raise ValueError("value must be a WebSocket")
        self._websocket = value

    def _construct_message(self, data: str, event: Optional[str] = None) -> Message:
        """Constructs message payload.

        Args:
            data: The data payload.
            event: The event name.
        """
        return dict(data=data, event=event)


class TokenWebSocketCallbackHandler(WebSocketCallbackHandler):
    """Callback handler for sending tokens in websocket sessions."""

    def __init__(self, *, output_key: str, **kwargs: dict[str, Any]) -> None:
        """Constructor method.

        Args:
            output_key: chain output key.
            **kwargs: Keyword arguments to pass to the parent constructor.
        """
        super().__init__(**kwargs)

        self.output_key = output_key

    async def on_chain_start(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Run when chain starts running."""
        self.streaming = False

    async def on_llm_new_token(self, token: str, **kwargs: dict[str, Any]) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if not self.streaming:
            self.streaming = True

        if self.llm_cache_used:  # cache missed (or was never enabled) if we are here
            self.llm_cache_used = False

        message = self._construct_message(
            data=get_token_data(token, self.mode), event="completion"
        )
        await self.websocket.send_json(message)

    async def on_chain_end(
        self, outputs: dict[str, Any], **kwargs: dict[str, Any]
    ) -> None:
        """Run when chain ends running.

        Final output is streamed only if LLM cache is enabled.
        """
        if self.llm_cache_used or not self.streaming:
            if self.output_key in outputs:
                message = self._construct_message(
                    data=get_token_data(outputs[self.output_key], self.mode),
                    event="completion",
                )
                await self.websocket.send_json(message)
            else:
                raise KeyError(f"missing outputs key: {self.output_key}")


class SourceDocumentsWebSocketCallbackHandler(WebSocketCallbackHandler):
    """Callback handler for sending source documents in websocket sessions."""

    async def on_chain_end(
        self, outputs: dict[str, Any], **kwargs: dict[str, Any]
    ) -> None:
        """Run when chain ends running."""
        if "source_documents" in outputs:
            if not isinstance(outputs["source_documents"], list):
                raise ValueError("source_documents must be a list")
            if not isinstance(outputs["source_documents"][0], Document):
                raise ValueError("source_documents must be a list of Document")

            # NOTE: langchain is using pydantic_v1 for `Document`
            source_documents: list[dict] = [
                document.dict() for document in outputs["source_documents"]
            ]
            message = self._construct_message(
                data=model_dump_json(
                    SourceDocumentsEventData(source_documents=source_documents)
                ),
                event=LangchainEvents.SOURCE_DOCUMENTS,
            )
            await self.websocket.send_json(message)


class FinalTokenWebSocketCallbackHandler(
    TokenWebSocketCallbackHandler, FinalStreamingStdOutCallbackHandler
):
    """Callback handler for sending final answer tokens in websocket sessions.

    Useful for streaming responses from Langchain Agents.
    """

    def __init__(
        self,
        *,
        answer_prefix_tokens: Optional[list[str]] = None,
        strip_tokens: bool = True,
        stream_prefix: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method.

        Args:
            answer_prefix_tokens: The answer prefix tokens to use.
            strip_tokens: Whether to strip tokens.
            stream_prefix: Whether to stream the answer prefix.
            **kwargs: Keyword arguments to pass to the parent constructor.
        """
        super().__init__(output_key=None, **kwargs)

        FinalStreamingStdOutCallbackHandler.__init__(
            self,
            answer_prefix_tokens=answer_prefix_tokens,
            strip_tokens=strip_tokens,
            stream_prefix=stream_prefix,
        )

    async def on_llm_start(self, *args, **kwargs) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False
        self.streaming = False

    async def on_llm_new_token(self, token: str, **kwargs: dict[str, Any]) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if not self.streaming:
            self.streaming = True

        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                message = self._construct_message(
                    data=get_token_data("".join(self.last_tokens), self.mode),
                    event="completion",
                )
                await self.websocket.send_json(message)

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            message = self._construct_message(
                data=get_token_data(token, self.mode), event="completion"
            )
            await self.websocket.send_json(message)