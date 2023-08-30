import time
from typing import Any, Callable, List
from swarms.models.prompts.agent_prompt_generator import get_prompt

class TokenUtils:
    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())


class PromptConstructor:
    def __init__(self, ai_name: str, ai_role: str, tools):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.tools = tools

    def construct_full_prompt(self, goals: List[str]) -> str:
        prompt_start = (
            """Your decisions must always be made independently 
            without seeking user assistance.\n
            Play to your strengths as an LLM and pursue simple 
            strategies with no legal complications.\n
            If you have completed all your tasks, make sure to 
            use the "finish" command."""
        )
        # Construct full prompt
        full_prompt = (
            f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(goals):
            full_prompt += f"{i+1}. {goal}\n"
        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt


class Message:
    content: str

    def count_tokens(self) -> int:
        return TokenUtils.count_tokens(self.content)

    def format_content(self) -> str:
        return self.content


class SystemMessage(Message):
    pass


class HumanMessage(Message):
    pass


class MessageFormatter:
    send_token_limit: int = 4196

    def format_messages(self, **kwargs: Any) -> List[Message]:
        prompt_constructor = PromptConstructor(ai_name=kwargs["ai_name"],
                                               ai_role=kwargs["ai_role"],
                                               tools=kwargs["tools"])
        base_prompt = SystemMessage(content=prompt_constructor.construct_full_prompt(kwargs["goals"]))
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}"
        )
        used_tokens = base_prompt.count_tokens() + time_prompt.count_tokens()
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [TokenUtils.count_tokens(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [TokenUtils.count_tokens(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += memory_message.count_tokens()
        historical_messages: List[Message] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = message.count_tokens()
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[Message] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
