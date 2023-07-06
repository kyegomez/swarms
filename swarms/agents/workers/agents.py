from __future__ import annotations
from enum import Enum
from typing import Callable, Tuple

from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import BaseTool, Tool


class ToolScope(Enum):
    GLOBAL = "global"
    SESSION = "session"

SessionGetter = Callable[[], Tuple[str, AgentExecutor]]


def tool(
    name: str,
    description: str,
    scope: ToolScope = ToolScope.GLOBAL,
):
    def decorator(func):
        func.name = name
        func.description = description
        func.is_tool = True
        func.scope = scope
        return func

    return decorator


class ToolWrapper:
    def __init__(self, name: str, description: str, scope: ToolScope, func):
        self.name = name
        self.description = description
        self.scope = scope
        self.func = func

    def is_global(self) -> bool:
        return self.scope == ToolScope.GLOBAL

    def is_per_session(self) -> bool:
        return self.scope == ToolScope.SESSION

    def to_tool(
        self,
        get_session: SessionGetter = lambda: [],
    ) -> BaseTool:
        func = self.func
        if self.is_per_session():
            func = lambda *args, **kwargs: self.func(
                *args, **kwargs, get_session=get_session
            )

        return Tool(
            name=self.name,
            description=self.description,
            func=func,
        )


class BaseToolSet:
    def tool_wrappers(cls) -> list[ToolWrapper]:
        methods = [
            getattr(cls, m) for m in dir(cls) if hasattr(getattr(cls, m), "is_tool")
        ]
        return [ToolWrapper(m.name, m.description, m.scope, m) for m in methods]


#=====================================>
from typing import Optional

from langchain.agents import load_tools
from langchain.agents.tools import BaseTool
from langchain.llms.base import BaseLLM


class ToolsFactory:
    @staticmethod
    def from_toolset(
        toolset: BaseToolSet,
        only_global: Optional[bool] = False,
        only_per_session: Optional[bool] = False,
        get_session: SessionGetter = lambda: [],
    ) -> list[BaseTool]:
        tools = []
        for wrapper in toolset.tool_wrappers():
            if only_global and not wrapper.is_global():
                continue
            if only_per_session and not wrapper.is_per_session():
                continue
            tools.append(wrapper.to_tool(get_session=get_session))
        return tools

    @staticmethod
    def create_global_tools(
        toolsets: list[BaseToolSet],
    ) -> list[BaseTool]:
        tools = []
        for toolset in toolsets:
            tools.extend(
                ToolsFactory.from_toolset(
                    toolset=toolset,
                    only_global=True,
                )
            )
        return tools

    @staticmethod
    def create_per_session_tools(
        toolsets: list[BaseToolSet],
        get_session: SessionGetter = lambda: [],
    ) -> list[BaseTool]:
        tools = []
        for toolset in toolsets:
            tools.extend(
                ToolsFactory.from_toolset(
                    toolset=toolset,
                    only_per_session=True,
                    get_session=get_session,
                )
            )
        return tools

    @staticmethod
    def create_global_tools_from_names(
        toolnames: list[str],
        llm: Optional[BaseLLM],
    ) -> list[BaseTool]:
        return load_tools(toolnames, llm=llm)

#=====================================>
#=====================================>





################
# from core.prompts.input import EVAL_PREFIX, EVAL_SUFFIX
from ...prompts.prompts import EVAL_PREFIX, EVAL_SUFFIX
############


from env import settings
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseOutputParser
from langchain.callbacks.base import BaseCallbackManager


class AgentBuilder:
    def __init__(self, toolsets: list[BaseToolSet] = []):
        self.llm: BaseChatModel = None
        self.parser: BaseOutputParser = None
        self.global_tools: list = None
        self.toolsets = toolsets

    def build_llm(self, callback_manager: BaseCallbackManager = None):
        self.llm = ChatOpenAI(
            temperature=0, callback_manager=callback_manager, verbose=True
        )
        self.llm.check_access()

    def build_parser(self):
        self.parser = EvalOutputParser()

    def build_global_tools(self):
        if self.llm is None:
            raise ValueError("LLM must be initialized before tools")

        toolnames = ["wikipedia"]

        if settings["SERPAPI_API_KEY"]:
            toolnames.append("serpapi")
        if settings["BING_SEARCH_URL"] and settings["BING_SUBSCRIPTION_KEY"]:
            toolnames.append("bing-search")

        self.global_tools = [
            *ToolsFactory.create_global_tools_from_names(toolnames, llm=self.llm),
            *ToolsFactory.create_global_tools(self.toolsets),
        ]

    def get_parser(self):
        if self.parser is None:
            raise ValueError("Parser is not initialized yet")

        return self.parser

    def get_global_tools(self):
        if self.global_tools is None:
            raise ValueError("Global tools are not initialized yet")

        return self.global_tools

    def get_agent(self):
        if self.llm is None:
            raise ValueError("LLM must be initialized before agent")

        if self.parser is None:
            raise ValueError("Parser must be initialized before agent")

        if self.global_tools is None:
            raise ValueError("Global tools must be initialized before agent")

        return ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=[
                *self.global_tools,
                *ToolsFactory.create_per_session_tools(
                    self.toolsets
                ),  # for names and descriptions
            ],
            system_message=EVAL_PREFIX.format(bot_name=settings["BOT_NAME"]),
            human_message=EVAL_SUFFIX.format(bot_name=settings["BOT_NAME"]),
            output_parser=self.parser,
            max_iterations=30,
        )
#===================================> callback

#===================================> callback

from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from celery import Task

from swarms.utils.utils import ANSI, Color, Style, dim_multiline, logger


class EVALCallbackHandler(BaseCallbackHandler):
    @property
    def ignore_llm(self) -> bool:
        return False

    def set_parser(self, parser) -> None:
        self.parser = parser

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        text = response.generations[0][0].text

        parsed = self.parser.parse_all(text)

        logger.info(ANSI("Plan").to(Color.blue().bright()) + ": " + parsed["plan"])
        logger.info(ANSI("What I Did").to(Color.blue()) + ": " + parsed["what_i_did"])
        logger.info(
            ANSI("Action").to(Color.cyan())
            + ": "
            + ANSI(parsed["action"]).to(Style.bold())
        )
        logger.info(
            ANSI("Input").to(Color.cyan())
            + ": "
            + dim_multiline(parsed["action_input"])
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        logger.info(ANSI(f"on_llm_new_token {token}").to(Color.green(), Style.italic()))

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        logger.info(ANSI(f"Entering new chain.").to(Color.green(), Style.italic()))
        logger.info(ANSI("Prompted Text").to(Color.yellow()) + f': {inputs["input"]}\n')

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        logger.info(ANSI(f"Finished chain.").to(Color.green(), Style.italic()))

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        logger.error(
            ANSI(f"Chain Error").to(Color.red()) + ": " + dim_multiline(str(error))
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            ANSI("Observation").to(Color.magenta()) + ": " + dim_multiline(output)
        )
        logger.info(ANSI("Thinking...").to(Color.green(), Style.italic()))

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        logger.error(ANSI("Tool Error").to(Color.red()) + f": {error}")

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        logger.info(
            ANSI("Final Answer").to(Color.yellow())
            + ": "
            + dim_multiline(finish.return_values.get("output", ""))
        )


class ExecutionTracingCallbackHandler(BaseCallbackHandler):
    def __init__(self, execution: Task):
        self.execution = execution
        self.index = 0

    def set_parser(self, parser) -> None:
        self.parser = parser

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        text = response.generations[0][0].text
        parsed = self.parser.parse_all(text)
        self.index += 1
        parsed["index"] = self.index
        self.execution.update_state(state="LLM_END", meta=parsed)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.execution.update_state(state="CHAIN_ERROR", meta={"error": str(error)})

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        previous = self.execution.AsyncResult(self.execution.request.id)
        self.execution.update_state(
            state="TOOL_END", meta={**previous.info, "observation": output}
        )

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        previous = self.execution.AsyncResult(self.execution.request.id)
        self.execution.update_state(
            state="TOOL_ERROR", meta={**previous.info, "error": str(error)}
        )

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        pass

#===================================> callback
#===================================> callback


#===================================> callback

from typing import Any, List, Optional, Sequence, Tuple

from langchain.agents.agent import Agent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AgentAction,
    AIMessage,
    BaseLanguageModel,
    BaseMessage,
    HumanMessage,
)
from langchain.tools.base import BaseTool

# from core.prompts.input import EVAL_TOOL_RESPONSE
from prompts.prompts import EVAL_TOOL_RESPONSE

from prompts.prompts import EVAL_FORMAT_INSTRUCTIONS


class ConversationalChatAgent(Agent):
    """An agent designed to hold a conversation in addition to using tools."""

    output_parser: BaseOutputParser

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought: "

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str,
        human_message: str,
        output_parser: BaseOutputParser,
        input_variables: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = human_message.format(
            format_instructions=output_parser.get_format_instructions()
        )
        final_prompt = format_instructions.format(
            tool_names=tool_names, tools=tool_strings
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        try:
            response = self.output_parser.parse(llm_output)
            return response["action"], response["action_input"]
        except Exception:
            raise ValueError(f"Could not parse LLM output: {llm_output}")

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            human_message = HumanMessage(
                content=EVAL_TOOL_RESPONSE.format(observation=observation)
            )
            thoughts.append(human_message)
        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        system_message: str,
        human_message: str,
        output_parser: BaseOutputParser,
        callback_manager: Optional[BaseCallbackManager] = None,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
            output_parser=output_parser,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=output_parser,
            **kwargs,
        )
#===================================> CHAT AGENT



#####===============================>

"""OpenAI chat wrapper."""

import logging
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import openai

from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env
from logger import logger
from pydantic import BaseModel, Extra, Field, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from env import settings



def _create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


async def acompletion_with_retry(llm: ChatOpenAI, **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict["content"])
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
    generations = []
    for res in response["choices"]:
        message = _convert_dict_to_message(res["message"])
        gen = ChatGeneration(message=message)
        generations.append(gen)
    return ChatResult(generations=generations)


class ModelNotFoundException(Exception):
    """Exception raised when the model is not found."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(
            f"\n\nModel {ANSI(self.model_name).to(Color.red())} does not exist.\nMake sure if you have access to the model.\n"
            + f"You can set the model name with the environment variable {ANSI('MODEL_NAME').to(Style.bold())} on {ANSI('.env').to(Style.bold())}.\n"
            + "\nex) MODEL_NAME=gpt-4\n"
            + ANSI(
                "\nLooks like you don't have access to gpt-4 yet. Try using `gpt-3.5-turbo`."
                if self.model_name == "gpt-4"
                else ""
            ).to(Style.italic())
        )


class ChatOpenAI(BaseChatModel, BaseModel):
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

    client: Any  #: :meta private:
    model_name: str = settings["MODEL_NAME"]
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: int = 2048
    """Maximum number of tokens to generate."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    def check_access(self) -> None:
        """Check that the user has access to the model."""

        try:
            openai.Engine.retrieve(self.model_name)
        except openai.error.InvalidRequestError:
            raise ModelNotFoundException(self.model_name)

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            **self.model_kwargs,
        }

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        import openai

        min_seconds = 4
        max_seconds = 10
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            response = self.client.create(**kwargs)
            logger.debug("Response:\n\t%s", response)
            return response

        return _completion_with_retry(**kwargs)

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        logger.debug("Messages:\n")
        for item in message_dicts:
            for k, v in item.items():
                logger.debug(f"\t\t{k}: {v}")
            logger.debug("\t-------")
        logger.debug("===========")

        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            for stream_resp in self.completion_with_retry(
                messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                self.callback_manager.on_llm_new_token(
                    token,
                    verbose=self.verbose,
                )
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        response = self.completion_with_retry(messages=message_dicts, **params)
        return _create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(
                self, messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                if self.callback_manager.is_async:
                    await self.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.verbose,
                    )
                else:
                    self.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.verbose,
                    )
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        else:
            response = await acompletion_with_retry(
                self, messages=message_dicts, **params
            )
            return _create_chat_result(response)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        # tiktoken NOT supported for Python 3.8 or below
        if sys.version_info[1] <= 8:
            return super().get_num_tokens(text)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3.5-Turbo encoder instance
        enc = tiktoken.encoding_for_model(self.model_name)

        # encode the text using the GPT-3.5-Turbo encoder
        tokenized_text = enc.encode(text)

        # calculate the number of tokens in the encoded text
        return len(tokenized_text)


###############LLM END =>



from typing import Dict, Optional

from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import CallbackManager
from langchain.callbacks import set_handler
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory

# from core.tools.base import BaseToolSet
# from core.tools.factory import ToolsFactory

# from .builder import AgentBuilder
# from .callback import EVALCallbackHandler, ExecutionTracingCallbackHandler


set_handler(EVALCallbackHandler())


class AgentManager:
    def __init__(
        self,
        toolsets: list[BaseToolSet] = [],
    ):
        self.toolsets: list[BaseToolSet] = toolsets
        self.memories: Dict[str, BaseChatMemory] = {}
        self.executors: Dict[str, AgentExecutor] = {}

    def create_memory(self) -> BaseChatMemory:
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_or_create_memory(self, session: str) -> BaseChatMemory:
        if not (session in self.memories):
            self.memories[session] = self.create_memory()
        return self.memories[session]

    def create_executor(
        self, session: str, execution: Optional[Task] = None
    ) -> AgentExecutor:
        builder = AgentBuilder(self.toolsets)
        builder.build_parser()

        callbacks = []
        eval_callback = EVALCallbackHandler()
        eval_callback.set_parser(builder.get_parser())
        callbacks.append(eval_callback)
        if execution:
            execution_callback = ExecutionTracingCallbackHandler(execution)
            execution_callback.set_parser(builder.get_parser())
            callbacks.append(execution_callback)

        callback_manager = CallbackManager(callbacks)

        builder.build_llm(callback_manager)
        builder.build_global_tools()

        memory: BaseChatMemory = self.get_or_create_memory(session)
        tools = [
            *builder.get_global_tools(),
            *ToolsFactory.create_per_session_tools(
                self.toolsets,
                get_session=lambda: (session, self.executors[session]),
            ),
        ]

        for tool in tools:
            tool.callback_manager = callback_manager

        executor = AgentExecutor.from_agent_and_tools(
            agent=builder.get_agent(),
            tools=tools,
            memory=memory,
            callback_manager=callback_manager,
            verbose=True,
        )
        self.executors[session] = executor
        return executor

    @staticmethod
    def create(toolsets: list[BaseToolSet]) -> "AgentManager":
        return AgentManager(
            toolsets=toolsets,
        )
    





#PARSER=============================+> PARSER
import re
from typing import Dict

from langchain.schema import BaseOutputParser

# from core.prompts.input import EVAL_FORMAT_INSTRUCTIONS

# from prompts.prompts import EVAL_FORMAT_INSTRUCTIONS
from ...prompts.prompts import EVAL_FORMAT_INSTRUCTIONS


class EvalOutputParser(BaseOutputParser):
    @staticmethod
    def parse_all(text: str) -> Dict[str, str]:
        regex = r"Action: (.*?)[\n]Plan:(.*)[\n]What I Did:(.*)[\n]Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise Exception("parse error")

        action = match.group(1).strip()
        plan = match.group(2)
        what_i_did = match.group(3)
        action_input = match.group(4).strip(" ")

        return {
            "action": action,
            "plan": plan,
            "what_i_did": what_i_did,
            "action_input": action_input,
        }

    def get_format_instructions(self) -> str:
        return EVAL_FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Dict[str, str]:
        regex = r"Action: (.*?)[\n]Plan:(.*)[\n]What I Did:(.*)[\n]Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise Exception("parse error")

        parsed = EvalOutputParser.parse_all(text)

        return {"action": parsed["action"], "action_input": parsed["action_input"]}

    def __str__(self):
        return "EvalOutputParser"

#PARSER=============================+> PARSER
