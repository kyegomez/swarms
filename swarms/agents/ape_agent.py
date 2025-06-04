from typing import Callable


from swarms.prompts.prompt_generator_optimizer import (
    OPENAI_PROMPT_GENERATOR_SYS_PROMPT,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="ape_agent")


def auto_generate_prompt(
    task: str = None,
    model: Callable = None,
    *args,
    **kwargs,
):
    """
    Generates a prompt for a given task using the provided model.

    Args:
    task (str, optional): The task for which to generate a prompt.
    model (Any, optional): The model to be used for prompt generation.
    max_tokens (int, optional): The maximum number of tokens in the generated prompt. Defaults to 4000.
    use_second_sys_prompt (bool, optional): Whether to use the second system prompt. Defaults to True.

    Returns:
    str: The generated prompt.
    """
    try:
        return model.run(
            task=f"{OPENAI_PROMPT_GENERATOR_SYS_PROMPT} \n\n Task: {task}"
        )
    except Exception as e:
        logger.error(f"Error generating prompt: {str(e)}")
        raise
