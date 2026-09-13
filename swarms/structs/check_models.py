import time
from typing import Any, Dict, Iterable, List, Optional

import httpx
from litellm import model_list
from loguru import logger

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CACHE_TTL_SECONDS = 300
OPENROUTER_TIMEOUT_SECONDS = 10
_BASE_MODELS = list(dict.fromkeys(model_list))
_BASE_MODEL_SET = frozenset(_BASE_MODELS)

_openrouter_models_cache = []
_openrouter_cache_expires_at = 0.0


def _parse_openrouter_models(payload: Dict[str, Any]) -> List[str]:
    """Turn an OpenRouter ``/models`` response into ``openrouter/<id>`` names."""
    return [
        f"openrouter/{model['id']}"
        for model in payload.get("data", [])
        if model.get("id")
    ]


def _store_openrouter_models(models: List[str]) -> None:
    global _openrouter_models_cache, _openrouter_cache_expires_at

    _openrouter_models_cache = models
    # Set expiry even on failure so a dead endpoint retries once a TTL.
    _openrouter_cache_expires_at = (
        time.monotonic() + OPENROUTER_CACHE_TTL_SECONDS
    )


def _openrouter_cache_is_fresh() -> bool:
    return time.monotonic() < _openrouter_cache_expires_at


def clear_openrouter_cache() -> None:
    """Forget the cached OpenRouter list so the next call fetches again."""
    global _openrouter_models_cache, _openrouter_cache_expires_at

    _openrouter_models_cache = []
    _openrouter_cache_expires_at = 0.0


def fetch_openrouter_models() -> List[str]:
    """
    Fetch the list of models from the OpenRouter API.

    Each model id is prefixed with ``openrouter/`` (e.g.
    ``openrouter/moonshotai/kimi-k3``). Results are cached for
    ``OPENROUTER_CACHE_TTL_SECONDS``.

    Returns:
        List[str]: The OpenRouter model names. Empty if the request fails.
    """
    if _openrouter_cache_is_fresh():
        return _openrouter_models_cache

    models = []
    try:
        with httpx.Client(
            timeout=OPENROUTER_TIMEOUT_SECONDS
        ) as client:
            response = client.get(OPENROUTER_MODELS_URL)
        response.raise_for_status()
        models = _parse_openrouter_models(response.json())
    except Exception as e:
        logger.warning(f"Could not fetch OpenRouter models: {str(e)}")

    _store_openrouter_models(models)
    return models


async def afetch_openrouter_models() -> List[str]:
    """Async form of :func:`fetch_openrouter_models`, sharing its cache."""
    if _openrouter_cache_is_fresh():
        return _openrouter_models_cache

    models: List[str] = []
    try:
        async with httpx.AsyncClient(
            timeout=OPENROUTER_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(OPENROUTER_MODELS_URL)
        response.raise_for_status()
        models = _parse_openrouter_models(response.json())
    except Exception as e:
        logger.warning(f"Could not fetch OpenRouter models: {str(e)}")

    _store_openrouter_models(models)
    return models


def _merge_models(
    openrouter_models: Iterable[str],
    exclude_keywords: Iterable[str],
) -> List[str]:
    """
    Merge OpenRouter models into the static litellm list.

    A model already present via litellm, with or without the ``openrouter/``
    prefix, is skipped so each model appears once. ``exclude_keywords`` drops
    any model whose name contains one of them, case-insensitively.
    """
    keywords = [keyword.lower() for keyword in exclude_keywords]

    merged = _BASE_MODELS + [
        model
        for model in dict.fromkeys(openrouter_models)
        if model not in _BASE_MODEL_SET
        and model.removeprefix("openrouter/") not in _BASE_MODEL_SET
    ]

    if not keywords:
        return merged

    return [
        model
        for model in merged
        if not any(keyword in model.lower() for keyword in keywords)
    ]


def _report(models: List[str]) -> Dict[str, Any]:
    return {
        "status": "success",
        "count": len(models),
        "models": models,
    }


def get_available_models(
    include_openrouter: bool = True,
    exclude_keywords: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    List every model name litellm knows, plus OpenRouter's live catalogue.

    Args:
        include_openrouter (bool): Fetch OpenRouter's model list (cached for
            5 minutes) and include it with an ``openrouter/`` prefix.
            Defaults to True.
        exclude_keywords (Iterable[str], optional): Drop any model whose name
            contains one of these, case-insensitively. Defaults to none.

    Returns:
        Dict[str, Any]: ``status`` ("success"), ``count`` and ``models``.
    """
    openrouter_models = (
        fetch_openrouter_models() if include_openrouter else []
    )
    return _report(
        _merge_models(openrouter_models, exclude_keywords or ())
    )


async def aget_available_models(
    include_openrouter: bool = True,
    exclude_keywords: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Async form of :func:`get_available_models`."""
    openrouter_models = (
        await afetch_openrouter_models() if include_openrouter else []
    )
    return _report(
        _merge_models(openrouter_models, exclude_keywords or ())
    )


def is_model_available(
    model: str, include_openrouter: bool = True
) -> bool:
    """
    Whether ``model`` is a name litellm or OpenRouter can serve.

    Args:
        model (str): A model name as passed to ``Agent(model_name=...)``.
        include_openrouter (bool): Also check OpenRouter's catalogue.
            Defaults to True.

    Returns:
        bool: True if the name is listed.
    """
    return model in get_available_models(include_openrouter)["models"]


def model_count(
    include_openrouter: bool = True,
    exclude_keywords: Optional[Iterable[str]] = None,
) -> int:
    """
    How many model names :func:`get_available_models` would return.

    Args:
        include_openrouter (bool): Count OpenRouter's catalogue too.
            Defaults to True.
        exclude_keywords (Iterable[str], optional): Drop any model whose name
            contains one of these, case-insensitively. Defaults to none.

    Returns:
        int: The number of available model names.
    """
    return get_available_models(include_openrouter, exclude_keywords)[
        "count"
    ]
