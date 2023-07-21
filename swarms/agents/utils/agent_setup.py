import os

from swarms.agents.prompts.prompts import EVAL_PREFIX, EVAL_SUFFIX
from swarms.agents.tools.main import BaseToolSet
from swarms.agents.tools.main import ToolsFactory


from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseOutputParser
from langchain.callbacks.base import BaseCallbackManager

from .ConversationalChatAgent import ConversationalChatAgent
# from .ChatOpenAI import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from .output_parser import EvalOutputParser



class AgentSetup:
    def __init__(self, toolsets: list[BaseToolSet] = [], openai_api_key: str = None, serpapi_api_key: str = None, bing_search_url: str = None, bing_subscription_key: str = None):
        self.llm: BaseChatModel = None
        self.parser: BaseOutputParser = None
        self.global_tools: list = None
        self.toolsets = toolsets
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.serpapi_api_key = serpapi_api_key or os.getenv('SERPAPI_API_KEY')
        self.bing_search_url = bing_search_url or os.getenv('BING_SEARCH_URL')
        self.bing_subscription_key = bing_subscription_key or os.getenv('BING_SUBSCRIPTION_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI key is missing, it should either be set as an environment variable or passed as a parameter")

    def setup_llm(self, callback_manager: BaseCallbackManager = None, openai_api_key: str = None):
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key is None:
                raise ValueError("OpenAI API key is missing. It should either be set as an environment variable or passed as a parameter.")
        
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.5, callback_manager=callback_manager, verbose=True)

    def setup_parser(self):
        self.parser = EvalOutputParser()

    def setup_global_tools(self):
        if self.llm is None:
            raise ValueError("LLM must be initialized before tools")

        toolnames = ["wikipedia"]

        if self.serpapi_api_key:
            toolnames.append("serpapi")
        
        if self.bing_search_url and self.bing_subscription_key:
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
            system_message=EVAL_PREFIX.format(bot_name=os.environ["BOT_NAME"] or 'WorkerUltraNode'),
            human_message=EVAL_SUFFIX.format(bot_name=os.environ["BOT_NAME"] or 'WorkerUltraNode'),
            output_parser=self.parser,
            max_iterations=30,
        )