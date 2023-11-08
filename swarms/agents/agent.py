from __future__ import annotations

import json
import time
from typing import Any, Callable, List, Optional

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import (
    BaseChatPromptTemplate,)
from langchain.schema import (
    BaseChatMessageHistory,
    Document,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain_experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain_experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain_experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
    get_prompt,
)
from langchain_experimental.pydantic_v1 import BaseModel, ValidationError

# PROMPT
FINISH_NAME = "finish"


# This class has a metaclass conflict: both `BaseChatPromptTemplate` and `BaseModel`
# define a metaclass to use, and the two metaclasses attempt to define
# the same functions but in mutually-incompatible ways.
# It isn't clear how to resolve this, and this code predates mypy
# beginning to perform that check.
#
# Mypy errors:
# ```
# Definition of "__private_attributes__" in base class "BaseModel" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__repr_name__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__pretty__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__repr_str__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__rich_repr__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Metaclass conflict: the metaclass of a derived class must be
#   a (non-strict) subclass of the metaclasses of all its bases  [misc]
# ```
#
# TODO: look into refactoring this class in a way that avoids the mypy type errors
class AutoGPTPrompt(BaseChatPromptTemplate, BaseModel):  # type: ignore[misc]
    """Prompt for AutoGPT."""

    ai_name: str
    ai_role: str
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, goals: List[str]) -> str:
        prompt_start = ("Your decisions must always be made independently "
                        "without seeking user assistance.\n"
                        "Play to your strengths as an LLM and pursue simple "
                        "strategies with no legal complications.\n"
                        "If you have completed all your tasks, make sure to "
                        'use the "finish" command.')
        # Construct full prompt
        full_prompt = (
            f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(
            content=self.construct_full_prompt(kwargs["goals"]))
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}")
        used_tokens = self.token_counter(
            base_prompt.content) + self.token_counter(time_prompt.content)
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(
            str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory])
        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory])
        content_format = (
            f"This reminds you of these events from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages


class PromptGenerator:
    """A class for generating custom prompt strings.

    Does this based on constraints, commands, resources, and performance evaluations.
    """

    def __init__(self) -> None:
        """Initialize the PromptGenerator object.

        Starts with empty lists of constraints, commands, resources,
        and performance evaluations.
        """
        self.constraints: List[str] = []
        self.commands: List[BaseTool] = []
        self.resources: List[str] = []
        self.performance_evaluation: List[str] = []
        self.response_format = {
            "thoughts": {
                "text":
                    "thought",
                "reasoning":
                    "reasoning",
                "plan":
                    "- short bulleted\n- list that conveys\n- long-term plan",
                "criticism":
                    "constructive self-criticism",
                "speak":
                    "thoughts summary to say to user",
            },
            "command": {
                "name": "command name",
                "args": {
                    "arg name": "value"
                }
            },
        }

    def add_constraint(self, constraint: str) -> None:
        """
        Add a constraint to the constraints list.

        Args:
            constraint (str): The constraint to be added.
        """
        self.constraints.append(constraint)

    def add_tool(self, tool: BaseTool) -> None:
        self.commands.append(tool)

    def _generate_command_string(self, tool: BaseTool) -> str:
        output = f"{tool.name}: {tool.description}"
        output += f", args json schema: {json.dumps(tool.args)}"
        return output

    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Args:
            resource (str): The resource to be added.
        """
        self.resources.append(resource)

    def add_performance_evaluation(self, evaluation: str) -> None:
        """
        Add a performance evaluation item to the performance_evaluation list.

        Args:
            evaluation (str): The evaluation item to be added.
        """
        self.performance_evaluation.append(evaluation)

    def _generate_numbered_list(self,
                                items: list,
                                item_type: str = "list") -> str:
        """
        Generate a numbered list from given items based on the item_type.

        Args:
            items (list): A list of items to be numbered.
            item_type (str, optional): The type of items in the list.
                Defaults to 'list'.

        Returns:
            str: The formatted numbered list.
        """
        if item_type == "command":
            command_strings = [
                f"{i + 1}. {self._generate_command_string(item)}"
                for i, item in enumerate(items)
            ]
            finish_description = (
                "use this to signal that you have finished all your objectives")
            finish_args = ('"response": "final response to let '
                           'people know you have finished your objectives"')
            finish_string = (f"{len(items) + 1}. {FINISH_NAME}: "
                             f"{finish_description}, args: {finish_args}")
            return "\n".join(command_strings + [finish_string])
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    def generate_prompt_string(self) -> str:
        """Generate a prompt string.

        Returns:
            str: The generated prompt string.
        """
        formatted_response_format = json.dumps(self.response_format, indent=4)
        prompt_string = (
            f"Constraints:\n{self._generate_numbered_list(self.constraints)}\n\n"
            "Commands:\n"
            f"{self._generate_numbered_list(self.commands, item_type='command')}\n\n"
            f"Resources:\n{self._generate_numbered_list(self.resources)}\n\n"
            "Performance Evaluation:\n"
            f"{self._generate_numbered_list(self.performance_evaluation)}\n\n"
            "You should only respond in JSON format as described below "
            f"\nResponse Format: \n{formatted_response_format} "
            "\nEnsure the response can be parsed by Python json.loads")

        return prompt_string


def get_prompt(tools: List[BaseTool]) -> str:
    """Generates a prompt string.

    It includes various constraints, commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    prompt_generator.add_constraint(
        "~16000 word limit for short term memory. "
        "Your short term memory is short, "
        "so immediately save important information to files.")
    prompt_generator.add_constraint(
        "If you are unsure how you previously did something "
        "or want to recall past events, "
        "thinking about similar events will help you remember.")
    prompt_generator.add_constraint("No user assistance")
    prompt_generator.add_constraint(
        'Exclusively use the commands listed in double quotes e.g. "command name"'
    )

    # Add commands to the PromptGenerator object
    for tool in tools:
        prompt_generator.add_tool(tool)

    # Add resources to the PromptGenerator object
    prompt_generator.add_resource(
        "Internet access for searches and information gathering.")
    prompt_generator.add_resource("Long Term memory management.")
    prompt_generator.add_resource(
        "GPT-3.5 powered Agents for delegation of simple tasks.")
    prompt_generator.add_resource("File output.")

    # Add performance evaluations to the PromptGenerator object
    prompt_generator.add_performance_evaluation(
        "Continuously review and analyze your actions "
        "to ensure you are performing to the best of your abilities.")
    prompt_generator.add_performance_evaluation(
        "Constructively self-criticize your big-picture behavior constantly.")
    prompt_generator.add_performance_evaluation(
        "Reflect on past decisions and strategies to refine your approach.")
    prompt_generator.add_performance_evaluation(
        "Every command has a cost, so be smart and efficient. "
        "Aim to complete tasks in the least number of steps.")

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()

    return prompt_string


class AutoGPT:
    """
    AutoAgent:


    Args:




    """

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.chat_history_memory = chat_history_memory or ChatMessageHistory()

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ) -> AutoGPT:
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
            chat_history_memory=chat_history_memory,
        )

    def run(self, goals: List[str]) -> str:
        user_input = ("Determine which next command to use, "
                      "and respond using the format specified above:")
        # Interaction Loop
        loop_count = 0
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.chat_history_memory.messages,
                memory=self.memory,
                user_input=user_input,
            )

            # Print Assistant thoughts
            print(assistant_reply)
            self.chat_history_memory.add_message(
                HumanMessage(content=user_input))
            self.chat_history_memory.add_message(
                AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return action.args["response"]
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    "Please refer to the 'COMMANDS' list for available "
                    "commands and only respond in the specified JSON format.")

            memory_to_add = f"Assistant Reply: {assistant_reply} \nResult: {result} "
            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += feedback

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.chat_history_memory.add_message(SystemMessage(content=result))
