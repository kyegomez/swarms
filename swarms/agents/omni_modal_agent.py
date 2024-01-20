
from langchain.base_language import BaseLanguageModel
from langchain_experimental.autonomous_agents.hugginggpt.repsonse_generator import (
    load_response_generator,
)
from langchain_experimental.autonomous_agents.hugginggpt.task_executor import (
    TaskExecutor,
)
from langchain_experimental.autonomous_agents.hugginggpt.task_planner import (
    load_chat_planner,
)
from transformers import load_tool

from swarms.agents.message import Message


class OmniModalAgent:
    """
    OmniModalAgent
    LLM -> Plans -> Tasks -> Tools -> Response

    Architecture:
    1. LLM: Language Model
    2. Chat Planner: Plans
    3. Task Executor: Tasks
    4. Tools: Tools

    Args:
        llm (BaseLanguageModel): Language Model
        tools (List[BaseTool]): List of tools

    Returns:
        str: response

    Usage:
    from swarms import OmniModalAgent, OpenAIChat,

    llm = OpenAIChat()
    agent = OmniModalAgent(llm)
    response = agent.run("Hello, how are you? Create an image of how your are doing!")
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        # tools: List[BaseTool]
    ):
        self.llm = llm

        print("Loading tools...")
        self.tools = [
            load_tool(tool_name)
            for tool_name in [
                "document-question-answering",
                "image-captioning",
                "image-question-answering",
                "image-segmentation",
                "speech-to-text",
                "summarization",
                "text-classification",
                "text-question-answering",
                "translation",
                "huggingface-tools/text-to-image",
                "huggingface-tools/text-to-video",
                "text-to-speech",
                "huggingface-tools/text-download",
                "huggingface-tools/image-transformation",
            ]
        ]

        self.chat_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        # self.task_executor = TaskExecutor
        self.history = []

    def run(self, input: str) -> str:
        """Run the OmniAgent"""
        plan = self.chat_planner.plan(
            inputs={
                "input": input,
                "hf_tools": self.tools,
            }
        )
        self.task_executor = TaskExecutor(plan)
        self.task_executor.run()

        response = self.response_generator.generate(
            {"task_execution": self.task_executor}
        )

        return response

    def chat(self, msg: str = None, streaming: bool = False):
        """
        Run chat

        Args:
            msg (str, optional): Message to send to the agent. Defaults to None.
            language (str, optional): Language to use. Defaults to None.
            streaming (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            str: Response from the agent

        Usage:
        --------------
        agent = MultiModalAgent()
        agent.chat("Hello")

        """

        # add users message to the history
        self.history.append(Message("User", msg))

        # process msg
        try:
            response = self.agent.run(msg)

            # add agent's response to the history
            self.history.append(Message("Agent", response))

            # if streaming is = True
            if streaming:
                return self._stream_response(response)
            else:
                response

        except Exception as error:
            error_message = f"Error processing message: {str(error)}"

            # add error to history
            self.history.append(Message("Agent", error_message))

            return error_message

    def _stream_response(self, response: str = None):
        """
        Yield the response token by token (word by word)

        Usage:
        --------------
        for token in _stream_response(response):
            print(token)

        """
        for token in response.split():
            yield token
