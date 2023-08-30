# from __future__ import annotations

# import logging
# from swarms.utils.logger import logger
# from typing import Any, Callable, Dict, List, Optional

# from pydantic import BaseModel, model_validator
# from tenacity import (
#     before_sleep_log,
#     retry,
#     retry_if_exception_type,
#     stop_after_attempt,
#     wait_exponential,
# )

# import google.generativeai as palm


# class GooglePalmError(Exception):
#     """Error raised when there is an issue with the Google PaLM API."""

# def _truncate_at_stop_tokens(
#     text: str,
#     stop: Optional[List[str]],
# ) -> str:
#     """Truncates text at the earliest stop token found."""
#     if stop is None:
#         return text

#     for stop_token in stop:
#         stop_token_idx = text.find(stop_token)
#         if stop_token_idx != -1:
#             text = text[:stop_token_idx]
#     return text


# def _response_to_result(response: palm.types.ChatResponse, stop: Optional[List[str]]) -> Dict[str, Any]:
#     """Convert a PaLM chat response to a result dictionary."""
#     result = {
#         "id": response.id,
#         "created": response.created,
#         "model": response.model,
#         "usage": {
#             "prompt_tokens": response.usage.prompt_tokens,
#             "completion_tokens": response.usage.completion_tokens,
#             "total_tokens": response.usage.total_tokens,
#         },
#         "choices": [],
#     }
#     for choice in response.choices:
#         result["choices"].append({
#             "text": _truncate_at_stop_tokens(choice.text, stop),
#             "index": choice.index,
#             "finish_reason": choice.finish_reason,
#         })
#     return result

# def _messages_to_prompt_dict(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """Convert a list of message dictionaries to a prompt dictionary."""
#     prompt = {"messages": []}
#     for message in messages:
#         prompt["messages"].append({
#             "role": message["role"],
#             "content": message["content"],
#         })
#     return prompt


# def _create_retry_decorator() -> Callable[[Any], Any]:
#     """Create a retry decorator with exponential backoff."""
#     return retry(
#         retry=retry_if_exception_type(GooglePalmError),
#         stop=stop_after_attempt(5),
#         wait=wait_exponential(multiplier=1, min=2, max=30),
#         before_sleep=before_sleep_log(logger, logging.DEBUG),
#         reraise=True,
#     )


# ####################### => main class
# class GooglePalm(BaseModel):
#     """Wrapper around Google's PaLM Chat API."""

#     client: Any  #: :meta private:
#     model_name: str = "models/chat-bison-001"
#     google_api_key: Optional[str] = None
#     temperature: Optional[float] = None
#     top_p: Optional[float] = None
#     top_k: Optional[int] = None
#     n: int = 1

#     @model_validator(mode="pre")
#     def validate_environment(cls, values: Dict) -> Dict:
#         # Same as before
#         pass

#     def chat_with_retry(self, **kwargs: Any) -> Any:
#         """Use tenacity to retry the completion call."""
#         retry_decorator = _create_retry_decorator()

#         @retry_decorator
#         def _chat_with_retry(**kwargs: Any) -> Any:
#             return self.client.chat(**kwargs)

#         return _chat_with_retry(**kwargs)

#     async def achat_with_retry(self, **kwargs: Any) -> Any:
#         """Use tenacity to retry the async completion call."""
#         retry_decorator = _create_retry_decorator()

#         @retry_decorator
#         async def _achat_with_retry(**kwargs: Any) -> Any:
#             return await self.client.chat_async(**kwargs)

#         return await _achat_with_retry(**kwargs)
    
#     def __call__(
#         self,
#         messages: List[Dict[str, Any]],
#         stop: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> Dict[str, Any]:
#         prompt = _messages_to_prompt_dict(messages)

#         response: palm.types.ChatResponse = self.chat_with_retry(
#             model=self.model_name,
#             prompt=prompt,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             top_k=self.top_k,
#             candidate_count=self.n,
#             **kwargs,
#         )

#         return _response_to_result(response, stop)

#     def generate(
#         self,
#         messages: List[Dict[str, Any]],
#         stop: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> Dict[str, Any]:
#         prompt = _messages_to_prompt_dict(messages)

#         response: palm.types.ChatResponse = self.chat_with_retry(
#             model=self.model_name,
#             prompt=prompt,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             top_k=self.top_k,
#             candidate_count=self.n,
#             **kwargs,
#         )

#         return _response_to_result(response, stop)

#     async def _agenerate(
#         self,
#         messages: List[Dict[str, Any]],
#         stop: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> Dict[str, Any]:
#         prompt = _messages_to_prompt_dict(messages)

#         response: palm.types.ChatResponse = await self.achat_with_retry(
#             model=self.model_name,
#             prompt=prompt,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             top_k=self.top_k,
#             candidate_count=self.n,
#         )

#         return _response_to_result(response, stop)

#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         """Get the identifying parameters."""
#         return {
#             "model_name": self.model_name,
#             "temperature": self.temperature,
#             "top_p": self.top_p,
#             "top_k": self.top_k,
#             "n": self.n,
#         }

#     @property
#     def _llm_type(self) -> str:
#         return "google-palm-chat"