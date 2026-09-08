from litellm import encode, model_list
from loguru import logger
from typing import Optional
from functools import lru_cache

# Use consistent default model
DEFAULT_MODEL = "gpt-5.4"


def count_tokens(
    text: str,
    model: str = DEFAULT_MODEL,
    default_encoder: Optional[str] = DEFAULT_MODEL,
) -> int:
    """
    Count the number of tokens in the given text using the specified model.

    Args:
        text: The text to tokenize
        model: The model to use for tokenization (defaults to gpt-4o-mini)
        default_encoder: Fallback encoder if the primary model fails (defaults to DEFAULT_MODEL)

    Returns:
        int: Number of tokens in the text

    Raises:
        ValueError: If text is empty or if both primary and fallback models fail
    """
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided")
        return 0

    # Set fallback encoder
    fallback_model = default_encoder or DEFAULT_MODEL

    # First attempt with the requested model
    try:
        tokens = encode(model=model, text=text)
        return len(tokens)

    except Exception as e:
        logger.warning(
            f"Failed to tokenize with model '{model}': {e} using fallback model '{fallback_model}'"
        )

        logger.info(f"Using fallback model '{fallback_model}'")

        # Only try fallback if it's different from the original model
        if fallback_model != model:
            try:
                logger.info(
                    f"Falling back to default encoder: {fallback_model}"
                )
                tokens = encode(model=fallback_model, text=text)
                return len(tokens)

            except Exception as fallback_error:
                logger.error(
                    f"Fallback encoder '{fallback_model}' also failed: {fallback_error}"
                )
                raise ValueError(
                    f"Both primary model '{model}' and fallback '{fallback_model}' failed to tokenize text"
                )
        else:
            logger.error(
                f"Primary model '{model}' failed and no different fallback available"
            )
            raise ValueError(
                f"Model '{model}' failed to tokenize text: {e}"
            )


@lru_cache(maxsize=100)
def get_supported_models() -> list:
    """Get list of supported models from litellm."""
    try:
        return model_list
    except Exception as e:
        logger.warning(f"Could not retrieve model list: {e}")
        return []


def cost_per_token(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> Optional[float]:
    """Dollar cost of one call at LiteLLM's listed price for ``model``.

    Args:
        model: The model name as passed to the agent.
        input_tokens: Prompt tokens, including any served from cache.
        output_tokens: Completion tokens, including reasoning.
        cached_tokens: The part of ``input_tokens`` served from cache, which
            most providers bill at a lower rate.

    Returns:
        The cost in USD, or None when LiteLLM has no price for the model.
    """
    from litellm import cost_per_token as _litellm_cost

    try:
        prompt_cost, completion_cost = _litellm_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cache_read_input_tokens=cached_tokens,
        )
    except Exception:
        return None
    return float(prompt_cost + completion_cost)
