from core.prompts.input import EVAL_PREFIX, EVAL_SUFFIX
from core.tools.base import BaseToolSet
from core.tools.factory import ToolsFactory
from env import settings
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseOutputParser
from langchain.callbacks.base import BaseCallbackManager

from .chat_agent import ConversationalChatAgent
from .llm import ChatOpenAI
from .parser import EvalOutputParser


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