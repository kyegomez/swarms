from typing import Union

from swarms.models.base_llm import AbstractLLM
from swarms.models.popular_llms import OpenAIChat
from swarms.prompts.meta_system_prompt import (
    meta_system_prompt_generator,
)
from swarms.structs.agent import Agent

meta_prompter_llm = OpenAIChat(system_prompt=str(meta_system_prompt_generator))


def meta_system_prompt(agent: Union[Agent, AbstractLLM], system_prompt: str) -> str:
    """
    Generates a meta system prompt for the given agent using the provided system prompt.

    Args:
        agent (Union[Agent, AbstractLLM]):
            The agent or LLM (Language Learning Model) for which the meta system prompt is generated.
        system_prompt (str):
            The system prompt used to generate the meta system prompt.

    Returns:
        str: The generated meta system prompt.
    """
    return meta_prompter_llm(system_prompt)
