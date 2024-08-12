from swarms.structs.agent import Agent
from typing import Union
from swarms.models.popular_llms import OpenAIChat
from swarms.models.base_llm import BaseLLM
from swarms.prompts.meta_system_prompt import (
    meta_system_prompt_generator,
)

meta_prompter_llm = OpenAIChat(
    system_prompt=str(meta_system_prompt_generator)
)


def meta_system_prompt(
    agent: Union[Agent, BaseLLM], system_prompt: str
) -> str:
    """
    Generates a meta system prompt for the given agent using the provided system prompt.

    Args:
        agent (Union[Agent, BaseLLM]): The agent or LLM (Language Learning Model) for which the meta system prompt is generated.
        system_prompt (str): The system prompt used to generate the meta system prompt.

    Returns:
        str: The generated meta system prompt.
    """
    return meta_prompter_llm(system_prompt)
