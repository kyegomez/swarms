from litellm import encode, model_list
from loguru import logger
from typing import Optional
from functools import lru_cache

# Use consistent default model
DEFAULT_MODEL = "gpt-4o-mini"


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


# if __name__ == "__main__":
#     # Test with different scenarios
#     test_text = "Hello, how are you?"

#     # # Test with Claude model
#     # try:
#     #     tokens = count_tokens(test_text, model="claude-3-5-sonnet-20240620")
#     #     print(f"Claude tokens: {tokens}")
#     # except Exception as e:
#     #     print(f"Claude test failed: {e}")

#     # # Test with default model
#     # try:
#     #     tokens = count_tokens(test_text)
#     #     print(f"Default model tokens: {tokens}")
#     # except Exception as e:
#     #     print(f"Default test failed: {e}")

#     # Test with explicit fallback
#     try:
#         tokens = count_tokens(test_text, model="some-invalid-model", default_encoder="gpt-4o-mini")
#         print(f"Fallback test tokens: {tokens}")
#     except Exception as e:
#         print(f"Fallback test failed: {e}")
