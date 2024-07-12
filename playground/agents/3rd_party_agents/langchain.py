from typing import List, Optional

from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import StringPromptTemplate
from langchain.tools import DuckDuckGoSearchRun

from swarms import Agent


class LangchainAgentWrapper(Agent):
    """
    Initialize the LangchainAgentWrapper.

    Args:
        name (str): The name of the agent.
        tools (List[Tool]): The list of tools available to the agent.
        llm (Optional[OpenAI], optional): The OpenAI language model to use. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        tools: List[Tool],
        llm: Optional[OpenAI] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.tools = tools
        self.llm = llm or OpenAI(temperature=0)

        prompt = StringPromptTemplate.from_template(
            "You are {name}, an AI assistant. Answer the following question: {question}"
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in self.tools]

        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=None,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def run(self, task: str, *args, **kwargs):
        """
        Run the agent with the given task.

        Args:
            task (str): The task to be performed by the agent.

        Returns:
            Any: The result of the agent's execution.
        """
        try:
            return self.agent_executor.run(task)
        except Exception as e:
            print(f"An error occurred: {e}")


# Usage example

search_tool = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Useful for searching the internet",
    )
]

langchain_wrapper = LangchainAgentWrapper("LangchainAssistant", tools)
result = langchain_wrapper.run("What is the capital of France?")
print(result)
