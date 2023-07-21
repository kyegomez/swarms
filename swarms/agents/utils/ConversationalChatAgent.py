from typing import Any, List, Optional, Sequence, Tuple
import logging

from swarms.agents.utils.Agent import Agent
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


from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction


from swarms.agents.prompts.prompts import EVAL_TOOL_RESPONSE



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationalChatAgent(Agent):
    """An agent designed to hold a conversation in addition to using tools."""

    output_parser: BaseOutputParser

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError
        
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """Get default output parser for this class."""

    
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
        if not isinstance(tools, Sequence):
            raise TypeError("Tools must be a sequence")
        if not isinstance(system_message, str):
            raise TypeError("System message must be a string")
        if not isinstance(human_message, str):
            raise TypeError("Human message must be a string")
        if not isinstance(output_parser, BaseOutputParser):
            raise TypeError("Output parser must be an instance of BaseOutputParser")
        if input_variables and not isinstance(input_variables, list):
            raise TypeError("Input variables must be a list")

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
        except Exception as e:
            logging.error(f"Error while extracting tool and input: {str(e)}")
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
        try:
            return cls(
                llm_chain=llm_chain,
                allowed_tools=tool_names,
                output_parser=output_parser,
                **kwargs,
            )
        except Exception as e:
            logging.error(f"Error while creating agent from LLM and tools: {str(e)}")
            raise e
    