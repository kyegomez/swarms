import traceback

from loguru import logger

from swarms.utils.litellm_tokenizer import count_tokens
from typing import Optional


def dynamic_auto_chunking_(
    content: str,
    context_length: Optional[int] = 8192,
    tokenizer_model_name: Optional[str] = "gpt-4.1",
):
    """
    Dynamically chunk the conversation history to fit within the context length.

    Args:
        content (str): The conversation history as a string.
        context_length (int): The maximum number of tokens allowed.
        tokenizer_model_name (str): The name of the tokenizer model to use.

    Returns:
        str: The chunked conversation history as a string that fits within context_length tokens.
    """
    total_tokens = count_tokens(
        text=content, model=tokenizer_model_name
    )

    if total_tokens <= context_length:
        return content

    # We need to remove characters from the beginning until we're under the limit
    # Start by removing a percentage of characters and adjust iteratively
    target_tokens = context_length
    current_string = content

    # Binary search approach to find the right cutoff point
    left, right = 0, len(content)

    while left < right:
        mid = (left + right) // 2
        test_string = content[mid:]

        if not test_string:
            break

        test_tokens = count_tokens(
            text=test_string, model=tokenizer_model_name
        )

        if test_tokens <= target_tokens:
            # We can remove more from the beginning
            right = mid
            current_string = test_string
        else:
            # We need to keep more from the beginning
            left = mid + 1

    return current_string


def dynamic_auto_chunking(
    content: str,
    context_length: Optional[int] = 8192,
    tokenizer_model_name: Optional[str] = "gpt-4.1",
):
    """
    Dynamically chunk the conversation history to fit within the context length.

    Args:
        content (str): The conversation history as a string.
        context_length (int): The maximum number of tokens allowed.
        tokenizer_model_name (str): The name of the tokenizer model to use.
    """
    try:
        return dynamic_auto_chunking_(
            content=content,
            context_length=context_length,
            tokenizer_model_name=tokenizer_model_name,
        )
    except Exception as e:
        logger.error(
            f"Dynamic auto chunking failed: {e} Traceback: {traceback.format_exc()}"
        )
        return content
