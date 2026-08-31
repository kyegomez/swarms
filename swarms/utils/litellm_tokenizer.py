import hashlib
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

from litellm import encode, model_list
from loguru import logger

# Use consistent default model
DEFAULT_MODEL = "gpt-5.4"

# count_tokens is called on the full, growing transcript on every turn from 22
# sites, so the same prefix is re-encoded continuously — the engine behind the
# O(n^2) conversation cost (#1735).
#
# Keyed by digest rather than by the text itself: an lru_cache over the raw
# string would pin a few hundred full transcripts in memory, which for long
# runs is worse than the work it saves. Only the 32-byte digest is retained.
_TOKEN_COUNT_CACHE: "OrderedDict[tuple, int]" = OrderedDict()
_TOKEN_COUNT_CACHE_MAX = 512


def _cache_key(text: str, model: str) -> tuple:
    return (
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        model,
    )


def _cache_get(key: tuple) -> Optional[int]:
    if key in _TOKEN_COUNT_CACHE:
        _TOKEN_COUNT_CACHE.move_to_end(key)
        return _TOKEN_COUNT_CACHE[key]
    return None


def _cache_put(key: tuple, value: int) -> None:
    _TOKEN_COUNT_CACHE[key] = value
    _TOKEN_COUNT_CACHE.move_to_end(key)
    while len(_TOKEN_COUNT_CACHE) > _TOKEN_COUNT_CACHE_MAX:
        _TOKEN_COUNT_CACHE.popitem(last=False)


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

    # litellm's encode is deterministic for a given model, so a hit is exact.
    # Only successful counts are cached; the fallback path below stays uncached
    # so a transient failure is never remembered as an answer.
    key = _cache_key(text, model)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # First attempt with the requested model
    try:
        tokens = encode(model=model, text=text)
        _cache_put(key, len(tokens))
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
