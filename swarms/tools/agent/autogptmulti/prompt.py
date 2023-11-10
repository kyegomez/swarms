import time
from typing import Any, Callable, List

from pydantic import BaseModel

from .prompt_generator import get_prompt
from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever


class AutoGPTPrompt(BaseChatPromptTemplate, BaseModel):
    ai_name: str
    ai_role: str
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, goals: List[str]) -> str:
        prompt_start = """Your decisions must always be made independently
            without seeking user assistance. Play to your strengths
            as an LLM and pursue simple strategies with no legal complications.
            Once you have completed your goal or have found that it can not be finished with current commands,
            make sure to use the "finish" command immediately."""

        # Construct full prompt
        full_prompt = (
            f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        )
        if isinstance(goals, list):
            for i, goal in enumerate(goals):
                full_prompt += f"{i+1}. {goal}\n"
        else:
            full_prompt += f"{goals}\n"
        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["goals"]))
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}"
        )
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]

        content_format = (
            f"This reminds you of these events "
            f"you have already used, and NEVER conduct repeated or unrelated commands:\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += len(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
