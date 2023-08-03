from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from pydantic import Field
from swarm.utils.logger import logger



######### helpers
if TYPE_CHECKING:
    import tiktoken


def _import_tiktoken() -> Any:
    try:
        import tiktoken
    except ImportError:
        raise ValueError(
            "Could not import tiktoken python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install tiktoken`."
        )
    return tiktoken

def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict

class BaseMessage:
    """Base message class."""


class HumanMessage(BaseMessage):
    """Human message class."""

    def __init__(self, content: str):
        self.role = "user"
        self.content = content


class AIMessage(BaseMessage):
    """AI message class."""

    def __init__(self, content: str, additional_kwargs: Optional[Dict[str, Any]] = None):
        self.role = "assistant"
        self.content = content
        self.additional_kwargs = additional_kwargs or {}


class SystemMessage(BaseMessage):
    """System message class."""

    def __init__(self, content: str):
        self.role = "system"
        self.content = content


class FunctionMessage(BaseMessage):
    """Function message class."""

    def __init__(self, content: str, name: str):
        self.role = "function"
        self.content = content
        self.name = name


class ChatMessage(BaseMessage):
    """Chat message class."""

    def __init__(self, content: str, role: str):
        self.role = role
        self.content = content


class BaseMessageChunk:
    """Base message chunk class."""


class HumanMessageChunk(BaseMessageChunk):
    """Human message chunk class."""

    def __init__(self, content: str):
        self.role = "user"
        self.content = content


class AIMessageChunk(BaseMessageChunk):
    """AI message chunk class."""

    def __init__(self, content: str, additional_kwargs: Optional[Dict[str, Any]] = None):
        self.role = "assistant"
        self.content = content
        self.additional_kwargs = additional_kwargs or {}


class SystemMessageChunk(BaseMessageChunk):
    """System message chunk class."""

    def __init__(self, content: str):
        self.role = "system"
        self.content = content


class FunctionMessageChunk(BaseMessageChunk):
    """Function message chunk class."""

    def __init__(self, content: str, name: str):
        self.role = "function"
        self.content = content
        self.name = name

class ChatMessageChunk(BaseMessageChunk):
    """Chat message chunk class."""

    def __init__(self, content: str, role: str):
        self.role = role
        self.content = content


def convert_openai_messages(messages: List[dict]) -> List[BaseMessage]:
    """Convert dictionaries representing OpenAI messages to LangChain format.

    Args:
        messages: List of dictionaries representing OpenAI messages

    Returns:
        List of LangChain BaseMessage objects.
    """
    converted_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if m.get("function_call"):
            additional_kwargs = {"function_call": dict(m["function_call"])}
        else:
            additional_kwargs = {}

        if role == "user":
            converted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            converted_messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
        elif role == "system":
            converted_messages.append(SystemMessage(content=content))
        elif role == "function":
            converted_messages.append(FunctionMessage(content=content, name=m["name"]))
        else:
            converted_messages.append(ChatMessage(content=content, role=role))

    return converted_messages


class ChatGenerationChunk:
    """Chat generation chunk class."""

    def __init__(self, message: BaseMessageChunk):
        self.message = message

    def __add__(self, other: "ChatGenerationChunk") -> "ChatGenerationChunk":
        if isinstance(self.message, AIMessageChunk) and isinstance(other.message, AIMessageChunk):
            combined_kwargs = {
                **self.message.additional_kwargs,
                **other.message.additional_kwargs,
            }
            return ChatGenerationChunk(
                AIMessageChunk(content=self.message.content + other.message.content, additional_kwargs=combined_kwargs)
            )
        return ChatGenerationChunk(BaseMessageChunk(content=self.message.content + other.message.content))

    @property
    def content(self) -> str:
        return self.message.content

    @property
    def additional_kwargs(self) -> Dict[str, Any]:
        return getattr(self.message, "additional_kwargs", {})


class ChatResult:
    """Chat result class."""

    def __init__(self, generations: List[ChatGenerationChunk]):
        self.generations = generations


class BaseChatModel:
    """Base chat model class."""

    def _convert_delta_to_message_chunk(self, _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]) -> BaseMessageChunk:
        role = _dict.get("role")
        content = _dict.get("content") or ""
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}

        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role == "function" or default_class == FunctionMessageChunk:
            return FunctionMessageChunk(content=content, name=_dict["name"])
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)
        else:
            return default_class(content=content)

    def _convert_dict_to_message(self, _dict: Mapping[str, Any]) -> BaseMessage:
        role = _dict["role"]
        if role == "user":
            return HumanMessage(content=_dict["content"])
        elif role == "assistant":
            content = _dict.get("content", "") or ""
            if _dict.get("function_call"):
                additional_kwargs = {"function_call": dict(_dict["function_call"])}
            else:
                additional_kwargs = {}
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif role == "system":
            return SystemMessage(content=_dict["content"])
        elif role == "function":
            return FunctionMessage(content=_dict["content"], name=_dict["name"])
        else:
            return ChatMessage(content=_dict["content"], role=role)

    def completion_with_retry(self, run_manager: Optional[Callable] = None, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)

    def _create_retry_decorator(
        self,
        llm: "ChatOpenAI",
        run_manager: Optional[Callable] = None,
    ) -> Callable[[Any], Any]:
        import openai

        errors = [
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
        ]
        return create_base_retry_decorator(
            error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
        )

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}


class ChatOpenAI(BaseChatModel):
    """Wrapper around OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatOpenAI
            openai = ChatOpenAI(model_name="gpt-3.5-turbo")
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        model_kwargs: Optional[Dict[str, Any]] = None,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        openai_organization: Optional[str] = None,
        openai_proxy: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        max_retries: int = 6,
        streaming: bool = False,
        n: int = 1,
        max_tokens: Optional[int] = None,
        tiktoken_model_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.model_kwargs = model_kwargs or {}
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.openai_organization = openai_organization
        self.openai_proxy = openai_proxy
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.streaming = streaming
        self.n = n
        self.max_tokens = max_tokens
        self.tiktoken_model_name = tiktoken_model_name

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the openai client."""
        openai_creds: Dict[str, Any] = {
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
            "organization": self.openai_organization,
            "model": self.model_name,
        }
        if self.openai_proxy:
            import openai

            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}
        return {**self._default_params, **openai_creds}

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai-chat"

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        tiktoken_ = _import_tiktoken()
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
            if model == "gpt-3.5-turbo":
                # gpt-3.5-turbo may change over time.
                # Returning num tokens assuming gpt-3.5-turbo-0301.
                model = "gpt-3.5-turbo-0301"
            elif model == "gpt-4":
                # gpt-4 may change over time.
                # Returning num tokens assuming gpt-4-0314.
                model = "gpt-4-0314"
        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken_.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken_.get_encoding(model)
        return model, encoding

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text with tiktoken package."""
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("gpt-3.5-turbo-0301"):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}."
                "See https://github.com/openai/openai-python/blob/main/chatml.md for "
                "information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens

    def get_token_usage(self, text: str, prefix_tokens: Optional[List[str]] = None) -> int:
        """Get the number of tokens used by the provided text."""
        token_ids = self.get_token_ids(text)
        if prefix_tokens:
            prefix_token_ids = self.get_token_ids(" ".join(prefix_tokens))
            token_ids = prefix_token_ids + token_ids
        return len(token_ids)

    def completion_with_retry(
        self, run_manager: Optional[Callable] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator(run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self, run_manager: Optional[Callable] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = self._create_retry_decorator(run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            # Use OpenAI's async api https://github.com/openai/openai-python#async-api
            return await self.client.acreate(**kwargs)

        return await _completion_with_retry(**kwargs)

    def _create_retry_decorator(
        self,
        run_manager: Optional[Callable] = None,
    ) -> Callable[[Any], Any]:
        import openai

        errors = [
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
        ]
        return create_base_retry_decorator(
            error_types=errors, max_retries=self.max_retries, run_manager=run_manager
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Callable] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__





















#=================================
# from typing import (
#     Any,
#     Dict,
#     List,
#     Mapping,
#     Optional,
# )

# import openai


# class ChatResult:
#     """Wrapper for the result of the chat generation process."""

#     def __init__(
#         self,
#         generations: List[ChatGeneration],
#         llm_output: Optional[Mapping[str, Any]] = None,
#     ):
#         self.generations = generations
#         self.llm_output = llm_output or {}

# class BaseMessage:
#     """Base class for different types of messages."""

#     def __init__(self, content: str):
#         self.content = content

# class AIMessage(BaseMessage):
#     """Message from the AI Assistant."""

#     def __init__(self, content: str, additional_kwargs: Optional[Dict[str, Any]] = None):
#         super().__init__(content)
#         self.additional_kwargs = additional_kwargs or {}

# class HumanMessage(BaseMessage):
#     """Message from the User."""

#     pass

# class SystemMessage(BaseMessage):
#     """System message."""

#     pass

# class FunctionMessage(BaseMessage):
#     """Function message."""

#     def __init__(self, content: str, name: str):
#         super().__init__(content)
#         self.name = name

# class ChatGeneration:
#     """Wrapper for the chat generation information."""

#     def __init__(
#         self, message: BaseMessage, generation_info: Optional[Mapping[str, Any]] = None
#     ):
#         self.message = message
#         self.generation_info = generation_info or {}

# class ChatGenerationChunk:
#     """Wrapper for a chunk of chat generation."""

#     def __init__(self, message: BaseMessage):
#         self.message = message

# def get_from_env_or_raise(var_name: str) -> str:
#     value = os.getenv(var_name)
#     if value is None:
#         raise ValueError(f"Environment variable {var_name} is not set.")
#     return value


# class OpenAI:
#     """Wrapper around OpenAI Chat large language models.

#     To use, you should have the ``openai`` python package installed, and the
#     environment variable ``OPENAI_API_KEY`` set with your API key.

#     Example:
#         .. code-block:: python

#             from langchain.chat_models import OpenAI
#             openai = OpenAI(model_name="gpt-3.5-turbo")
#     """

#     def __init__(
#         self,
#         model_name: str = "gpt-3.5-turbo",
#         temperature: float = 0.7,
#         openai_api_key: Optional[str] = None,
#         request_timeout: Optional[float] = None,
#         max_retries: int = 6,
#     ):
#         self.model_name = model_name
#         self.temperature = temperature
#         self.openai_api_key = openai_api_key
#         self.request_timeout = request_timeout
#         self.max_retries = max_retries
#         self._initialize_openai()

#     def _initialize_openai(self):
#         """Initialize the OpenAI client."""
#         if self.openai_api_key is None:
#             raise ValueError("OPENAI_API_KEY environment variable is not set.")

#         openai.api_key = self.openai_api_key

#     def _create_retry_decorator(self):
#         """Create a decorator to handle API call retries."""
#         errors = [
#             openai.error.Timeout,
#             openai.error.APIError,
#             openai.error.APIConnectionError,
#             openai.error.RateLimitError,
#             openai.error.ServiceUnavailableError,
#         ]

#         def retry_decorator(func):
#             @wraps(func)
#             def wrapper(*args, **kwargs):
#                 for _ in range(self.max_retries):
#                     try:
#                         return func(*args, **kwargs)
#                     except tuple(errors):
#                         continue
#                 raise ValueError("Max retries reached. Unable to complete the API call.")

#             return wrapper

#         return retry_decorator

#     def _create_message_dict(self, message: BaseMessage) -> Dict[str, Any]:
#         """Convert a LangChain message to an OpenAI message dictionary."""
#         role = message.role
#         content = message.content
#         message_dict = {"role": role, "content": content}

#         if role == "assistant" and isinstance(message, AIMessage):
#             message_dict["function_call"] = message.additional_kwargs.get("function_call", {})

#         if role == "function" and isinstance(message, FunctionMessage):
#             message_dict["name"] = message.name

#         return message_dict

#     def _create_message_dicts(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
#         """Convert a list of LangChain messages to a list of OpenAI message dictionaries."""
#         return [self._create_message_dict(message) for message in messages]

#     @retry_decorator
#     def _openai_completion(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Any:
#         """Call the OpenAI Chat Completion API."""
#         response = openai.ChatCompletion.create(messages=messages, **params)
#         return response

#     def generate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         """Generate a response using the OpenAI Chat model.

#         Args:
#             messages (List[BaseMessage]): List of messages in the conversation.
#             stop (Optional[List[str]]): List of stop sequences to stop generation.

#         Returns:
#             ChatResult: The generated response wrapped in ChatResult object.
#         """
#         params = {
#             "model": self.model_name,
#             "temperature": self.temperature,
#             "max_tokens": kwargs.get("max_tokens"),
#             "stream": kwargs.get("streaming", False),
#             "n": kwargs.get("n", 1),
#             "request_timeout": kwargs.get("request_timeout", self.request_timeout),
#         }

#         messages_dicts = self._create_message_dicts(messages)
#         response = self._openai_completion(messages_dicts, params)

#         # Process the response and create ChatResult
#         generations = []
#         for choice in response["choices"]:
#             message = self._convert_message(choice["message"])
#             generation_info = {"finish_reason": choice.get("finish_reason")}
#             generation = ChatGeneration(message=message, generation_info=generation_info)
#             generations.append(generation)

#         return ChatResult(generations=generations)

#     def _convert_message(self, message_dict: Dict[str, Any]) -> BaseMessage:
#         """Convert an OpenAI message dictionary to a LangChain message."""
#         role = message_dict["role"]
#         content = message_dict["content"]

#         if role == "user":
#             return HumanMessage(content=content)
#         elif role == "assistant":
#             additional_kwargs = message_dict.get("function_call", {})
#             return AIMessage(content=content, additional_kwargs=additional_kwargs)
#         elif role == "system":
#             return SystemMessage(content=content)
#         elif role == "function":
#             name = message_dict.get("name", "")
#             return FunctionMessage(content=content, name=name)
#         else:
#             raise ValueError(f"Invalid role found in the message: {role}")
