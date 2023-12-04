import contextlib
import datetime
import functools
import importlib
import re
import warnings
from importlib.metadata import version
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output import GenerationChunk
from langchain.schema.prompt import PromptValue
from langchain.utils import (
    check_package_version,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from packaging.version import parse
from requests import HTTPError, Response


def xor_args(*arg_groups: Tuple[str, ...]) -> Callable:
    """Validate specified keyword args are mutually exclusive."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Validate exactly one arg in each group is not None."""
            counts = [
                sum(
                    1
                    for arg in arg_group
                    if kwargs.get(arg) is not None
                )
                for arg_group in arg_groups
            ]
            invalid_groups = [
                i for i, count in enumerate(counts) if count != 1
            ]
            if invalid_groups:
                invalid_group_names = [
                    ", ".join(arg_groups[i]) for i in invalid_groups
                ]
                raise ValueError(
                    "Exactly one argument in each of the following"
                    " groups must be defined:"
                    f" {', '.join(invalid_group_names)}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def raise_for_status_with_text(response: Response) -> None:
    """Raise an error with the response text."""
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e


@contextlib.contextmanager
def mock_now(dt_value):  # type: ignore
    """Context manager for mocking out datetime.now() in unit tests.

    Example:
    with mock_now(datetime.datetime(2011, 2, 3, 10, 11)):
        assert datetime.datetime.now() == datetime.datetime(2011, 2, 3, 10, 11)
    """

    class MockDateTime(datetime.datetime):
        """Mock datetime.datetime.now() with a fixed datetime."""

        @classmethod
        def now(cls):  # type: ignore
            # Create a copy of dt_value.
            return datetime.datetime(
                dt_value.year,
                dt_value.month,
                dt_value.day,
                dt_value.hour,
                dt_value.minute,
                dt_value.second,
                dt_value.microsecond,
                dt_value.tzinfo,
            )

    real_datetime = datetime.datetime
    datetime.datetime = MockDateTime
    try:
        yield datetime.datetime
    finally:
        datetime.datetime = real_datetime


def guard_import(
    module_name: str,
    *,
    pip_name: Optional[str] = None,
    package: Optional[str] = None,
) -> Any:
    """Dynamically imports a module and raises a helpful exception if the module is not
    installed."""
    try:
        module = importlib.import_module(module_name, package)
    except ImportError:
        raise ImportError(
            f"Could not import {module_name} python package. Please"
            " install it with `pip install"
            f" {pip_name or module_name}`."
        )
    return module


def check_package_version(
    package: str,
    lt_version: Optional[str] = None,
    lte_version: Optional[str] = None,
    gt_version: Optional[str] = None,
    gte_version: Optional[str] = None,
) -> None:
    """Check the version of a package."""
    imported_version = parse(version(package))
    if lt_version is not None and imported_version >= parse(
        lt_version
    ):
        raise ValueError(
            f"Expected {package} version to be < {lt_version}."
            f" Received {imported_version}."
        )
    if lte_version is not None and imported_version > parse(
        lte_version
    ):
        raise ValueError(
            f"Expected {package} version to be <= {lte_version}."
            f" Received {imported_version}."
        )
    if gt_version is not None and imported_version <= parse(
        gt_version
    ):
        raise ValueError(
            f"Expected {package} version to be > {gt_version}."
            f" Received {imported_version}."
        )
    if gte_version is not None and imported_version < parse(
        gte_version
    ):
        raise ValueError(
            f"Expected {package} version to be >= {gte_version}."
            f" Received {imported_version}."
        )


def get_pydantic_field_names(pydantic_cls: Any) -> Set[str]:
    """Get field names, including aliases, for a pydantic class.

    Args:
        pydantic_cls: Pydantic class."""
    all_required_field_names = set()
    for field in pydantic_cls.__fields__.values():
        all_required_field_names.add(field.name)
        if field.has_alias:
            all_required_field_names.add(field.alias)
    return all_required_field_names


def build_extra_kwargs(
    extra_kwargs: Dict[str, Any],
    values: Dict[str, Any],
    all_required_field_names: Set[str],
) -> Dict[str, Any]:
    """Build extra kwargs from values and extra_kwargs.

    Args:
        extra_kwargs: Extra kwargs passed in by user.
        values: Values passed in by user.
        all_required_field_names: All required field names for the pydantic class.
    """
    for field_name in list(values):
        if field_name in extra_kwargs:
            raise ValueError(f"Found {field_name} supplied twice.")
        if field_name not in all_required_field_names:
            warnings.warn(
                f"""WARNING! {field_name} is not default parameter.
                {field_name} was transferred to model_kwargs.
                Please confirm that {field_name} is what you intended."""
            )
            extra_kwargs[field_name] = values.pop(field_name)

    invalid_model_kwargs = all_required_field_names.intersection(
        extra_kwargs.keys()
    )
    if invalid_model_kwargs:
        raise ValueError(
            f"Parameters {invalid_model_kwargs} should be specified"
            " explicitly. Instead they were passed in as part of"
            " `model_kwargs` parameter."
        )

    return extra_kwargs

class _AnthropicCommon(BaseLanguageModel):
    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: str ="claude-2"
    """Model name to use."""

    max_tokens_to_sample: int =256
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    streaming: bool = False
    """Whether to stream the results."""

    default_request_timeout: Optional[float] = None
    """Timeout for requests to Anthropic Completion API. Default is 600 seconds."""

    anthropic_api_url: Optional[str] = None

    anthropic_api_key: Optional[SecretStr] = None

    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None
    count_tokens: Optional[Callable[[str], int]] = None
    model_kwargs: Dict[str, Any] = {}

    @classmethod
    def build_extra(cls, values: Dict) -> Dict:
        extra = values.get("model_kwargs", {})
        all_required_field_names = get_pydantic_field_names(cls)
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["anthropic_api_key"] = get_from_dict_or_env(
                values, "anthropic_api_key", "ANTHROPIC_API_KEY"
        )
        # Get custom api url from environment.
        values["anthropic_api_url"] = get_from_dict_or_env(
            values,
            "anthropic_api_url",
            "ANTHROPIC_API_URL",
            default="https://api.anthropic.com",
        )

        try:
            import anthropic

            check_package_version("anthropic", gte_version="0.3")
            values["client"] = anthropic.Anthropic(
                base_url=values["anthropic_api_url"],
                api_key=values[
                    "anthropic_api_key"
                ].get_secret_value(),
                timeout=values["default_request_timeout"],
            )
            values["async_client"] = anthropic.AsyncAnthropic(
                base_url=values["anthropic_api_url"],
                api_key=values[
                    "anthropic_api_key"
                ].get_secret_value(),
                timeout=values["default_request_timeout"],
            )
            values["HUMAN_PROMPT"] = anthropic.HUMAN_PROMPT
            values["AI_PROMPT"] = anthropic.AI_PROMPT
            values["count_tokens"] = values["client"].count_tokens

        except ImportError:
            raise ImportError(
                "Could not import anthropic python package. "
                "Please it install it with `pip install anthropic`."
            )
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        d = {
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "model": self.model,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return {**d, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{}, **self._default_params}

    def _get_anthropic_stop(
        self, stop: Optional[List[str]] = None
    ) -> List[str]:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError(
                "Please ensure the anthropic package is loaded"
            )

        if stop is None:
            stop = []

        # Never want model to invent new turns of Human / Assistant dialog.
        stop.extend([self.HUMAN_PROMPT])

        return stop


class Anthropic(LLM, _AnthropicCommon):
    """Anthropic large language models.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            import anthropic
            from langchain.llms import Anthropic

            model = Anthropic(model="<model_name>", anthropic_api_key="my-api-key")

            # Simplest invocation, automatically wrapped with HUMAN_PROMPT
            # and AI_PROMPT.
            response = model("What are the biggest risks facing humanity?")

            # Or if you want to use the chat mode, build a few-shot-prompt, or
            # put words in the Assistant's mouth, use HUMAN_PROMPT and AI_PROMPT:
            raw_prompt = "What are the biggest risks facing humanity?"
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
            response = model(prompt)
    """
    
    @classmethod
    def raise_warning(cls, values: Dict) -> Dict:
        """Raise warning that this class is deprecated."""
        warnings.warn(
            "This Anthropic LLM is deprecated. Please use `from"
            " langchain.chat_models import ChatAnthropic` instead"
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "anthropic-llm"

    def _wrap_prompt(self, prompt: str) -> str:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError(
                "Please ensure the anthropic package is loaded"
            )

        if prompt.startswith(self.HUMAN_PROMPT):
            return prompt  # Already wrapped.

        # Guard against common errors in specifying wrong number of newlines.
        corrected_prompt, n_subs = re.subn(
            r"^\n*Human:", self.HUMAN_PROMPT, prompt
        )
        if n_subs == 1:
            return corrected_prompt

        # As a last resort, wrap the prompt ourselves to emulate instruct-style.
        return (
            f"{self.HUMAN_PROMPT} {prompt}{self.AI_PROMPT} Sure, here"
            " you go:\n"
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        r"""Call out to Anthropic's completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "What are the biggest risks facing humanity?"
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                response = model(prompt)

        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}
        response = self.client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **params,
        )
        return response.completion

    def convert_prompt(self, prompt: PromptValue) -> str:
        return self._wrap_prompt(prompt.to_string())

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Anthropic's completion endpoint asynchronously."""
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

        response = await self.async_client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **params,
        )
        return response.completion

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        r"""Call Anthropic completion_stream and return the resulting generator.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            A generator representing the stream of tokens from Anthropic.
        Example:
            .. code-block:: python

                prompt = "Write a poem about a stream."
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                generator = anthropic.stream(prompt)
                for token in generator:
                    yield token
        """
        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

        for token in self.client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            stream=True,
            **params,
        ):
            chunk = GenerationChunk(text=token.completion)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        r"""Call Anthropic completion_stream and return the resulting generator.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            A generator representing the stream of tokens from Anthropic.
        Example:
            .. code-block:: python
                prompt = "Write a poem about a stream."
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                generator = anthropic.stream(prompt)
                for token in generator:
                    yield token
        """
        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

        async for token in await self.async_client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            stream=True,
            **params,
        ):
            chunk = GenerationChunk(text=token.completion)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text, chunk=chunk
                )

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        if not self.count_tokens:
            raise NameError(
                "Please ensure the anthropic package is loaded"
            )
        return self.count_tokens(text)
