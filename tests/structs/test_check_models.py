import asyncio

import pytest

from swarms.structs import check_models
from swarms.structs.check_models import (
    afetch_openrouter_models,
    aget_available_models,
    clear_openrouter_cache,
    fetch_openrouter_models,
    get_available_models,
    is_model_available,
    model_count,
)


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self._data}


class _Client:
    """Stand-in for httpx.Client / httpx.AsyncClient."""

    calls = 0
    data = [{"id": "fake-lab/alpha"}, {"id": "fake-lab/beta"}, {}]

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url):
        _Client.calls += 1
        return _Response(_Client.data)


class _AsyncClient(_Client):
    async def get(self, url):
        _Client.calls += 1
        return _Response(_Client.data)


@pytest.fixture(autouse=True)
def fake_openrouter(monkeypatch):
    _Client.calls = 0
    monkeypatch.setattr(check_models.httpx, "Client", _Client)
    monkeypatch.setattr(
        check_models.httpx, "AsyncClient", _AsyncClient
    )
    clear_openrouter_cache()
    yield
    clear_openrouter_cache()


def test_fetch_prefixes_ids_and_skips_blank_entries():
    models = fetch_openrouter_models()

    assert models == [
        "openrouter/fake-lab/alpha",
        "openrouter/fake-lab/beta",
    ]


def test_fetch_is_cached_for_the_ttl():
    fetch_openrouter_models()
    fetch_openrouter_models()

    assert _Client.calls == 1


def test_async_fetch_shares_the_cache():
    fetch_openrouter_models()
    models = asyncio.run(afetch_openrouter_models())

    assert models[0] == "openrouter/fake-lab/alpha"
    assert _Client.calls == 1


def test_fetch_failure_returns_empty_and_does_not_raise(monkeypatch):
    def boom(self, url):
        raise RuntimeError("down")

    monkeypatch.setattr(_Client, "get", boom)

    assert fetch_openrouter_models() == []


def test_available_models_include_litellm_and_openrouter():
    report = get_available_models()

    assert report["status"] == "success"
    assert report["count"] == len(report["models"])
    assert "gpt-4o" in report["models"]
    assert "openrouter/fake-lab/alpha" in report["models"]


def test_available_models_without_openrouter_skips_the_fetch():
    report = get_available_models(include_openrouter=False)

    assert _Client.calls == 0
    # litellm's own list already carries some openrouter/ names; only the
    # live-fetched ones must be absent.
    assert "openrouter/fake-lab/alpha" not in report["models"]
    assert report["count"] == len(check_models._BASE_MODELS)


def test_openrouter_duplicates_of_litellm_models_are_dropped(
    monkeypatch,
):
    monkeypatch.setattr(
        _Client, "data", [{"id": "openai/gpt-4o"}, {"id": "gpt-4o"}]
    )
    report = get_available_models()

    models = report["models"]
    assert models.count("gpt-4o") == 1
    assert "openrouter/gpt-4o" not in models


def test_exclude_keywords_are_case_insensitive():
    report = get_available_models(exclude_keywords=["ALPHA", "gpt"])

    assert not any("alpha" in m.lower() for m in report["models"])
    assert not any("gpt" in m.lower() for m in report["models"])
    assert "openrouter/fake-lab/beta" in report["models"]


def test_async_report_matches_sync():
    sync = get_available_models()
    async_ = asyncio.run(aget_available_models())

    assert async_ == sync


def test_is_model_available():
    assert is_model_available("gpt-4o")
    assert is_model_available("openrouter/fake-lab/beta")
    assert not is_model_available("no-such-model")


def test_model_count_matches_report():
    assert model_count() == get_available_models()["count"]
    assert model_count(include_openrouter=False) == len(
        check_models._BASE_MODELS
    )
    assert model_count(exclude_keywords=["gpt"]) < model_count()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
