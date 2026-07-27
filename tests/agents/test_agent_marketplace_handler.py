"""
Test suite for :class:`swarms.agents.agent_marketplace_handler.AgentMarketplaceHandler`
and the agent error hierarchy in :mod:`swarms.schemas.agent_errors`.

Approach
--------
No network access and no real API key are used anywhere in this file. Every
HTTP interaction is served by ``httpx.MockTransport``, which keeps a real
``httpx.Client`` in the loop (real request construction, real URL encoding,
real status-code handling) while never touching a socket.

There are exactly two seams where the handler talks to the network, and both
are monkeypatched per-test to route through a ``MockTransport``:

* ``_client(timeout)`` — the module-level, ``lru_cache``d helper used by
  ``AgentMarketplaceHandler.fetch`` (GET). We replace the module attribute
  with a small factory that builds a real ``httpx.Client`` against a mock
  transport. Because the *real* ``_client`` is ``lru_cache``d, its cache is
  cleared before and after every test so no stale, un-mocked client leaks
  between tests.
* ``httpx.Client`` itself — used as a bare ``with httpx.Client(...) as
  client:`` context manager inside ``AgentMarketplaceHandler.add_prompt``
  (POST). We monkeypatch ``httpx.Client`` to a subclass that injects the mock
  transport, so the POST path still exercises real request/response
  machinery.

``SWARMS_API_KEY`` is set/unset per test with ``monkeypatch.setenv`` /
``monkeypatch.delenv`` — nothing is read from a real environment or file.

Constructing a real ``swarms.Agent`` works fully offline (no LLM call happens
until ``.run()``), so the ``Agent`` integration tests below build real agents
via ``Agent(agent_name=..., model_name="gpt-4o-mini", persistent_memory=False,
print_on=False)``. ``.run()`` is never called anywhere in this file.

Run:
    cd /Users/swarms_wd/Desktop/research/swarms
    PYTHONPATH=. python3 -m pytest tests/agents/test_agent_marketplace_handler.py -q -p no:randomly
"""

import json

import httpx
import pytest

from swarms import Agent
from swarms.agents.agent_marketplace_handler import (
    AgentMarketplaceHandler,
    DEFAULT_AGENT_NAME,
    DEFAULT_CATEGORY,
    MARKETPLACE_BASE_URL,
    _client,
)
from swarms.schemas import agent_errors as schema_errors
from swarms.schemas.agent_errors import AgentInitializationError
from swarms.structs import agent as agent_module

API_KEY = "test-swarms-api-key"


########################################################
# Fixtures: env, cache hygiene, mock transports
########################################################


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """The real ``_client`` is lru_cache'd; never let a stale client leak
    between tests, whether or not a given test replaces it."""
    _client.cache_clear()
    yield
    _client.cache_clear()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("SWARMS_API_KEY", API_KEY)
    return API_KEY


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("SWARMS_API_KEY", raising=False)


def install_get_transport(monkeypatch, handler):
    """Route the module-level ``_client(timeout)`` (used by GET/fetch)
    through a real ``httpx.Client`` backed by ``MockTransport``.

    Returns the list of timeouts the fake factory was called with, so tests
    can assert the ``timeout`` argument was honored end to end.
    """
    transport = httpx.MockTransport(handler)
    calls = []

    def fake_client(timeout):
        calls.append(timeout)
        return httpx.Client(transport=transport, timeout=timeout)

    monkeypatch.setattr(
        agent_marketplace_handler_module(), "_client", fake_client
    )
    return calls


def install_post_transport(monkeypatch, handler):
    """Route ``with httpx.Client(...) as client:`` (used by POST/add_prompt)
    through a real ``httpx.Client`` subclass backed by ``MockTransport``.
    """
    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.Client

    class _MockedClient(real_client_cls):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MockedClient)
    return transport


def agent_marketplace_handler_module():
    import swarms.agents.agent_marketplace_handler as mod

    return mod


def json_response(status_code=200, payload=None):
    return httpx.Response(status_code, json=payload or {})


########################################################
# check_api_key()
########################################################


class TestCheckApiKey:
    def test_returns_env_value(self, api_key):
        assert AgentMarketplaceHandler.check_api_key() == API_KEY

    def test_raises_when_unset(self, no_api_key):
        with pytest.raises(ValueError, match="SWARMS_API_KEY"):
            AgentMarketplaceHandler.check_api_key()

    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setenv("SWARMS_API_KEY", "")
        with pytest.raises(ValueError, match="SWARMS_API_KEY"):
            AgentMarketplaceHandler.check_api_key()

    def test_raises_when_whitespace_only(self, monkeypatch):
        monkeypatch.setenv("SWARMS_API_KEY", "   ")
        with pytest.raises(ValueError, match="SWARMS_API_KEY"):
            AgentMarketplaceHandler.check_api_key()

    def test_read_fresh_every_call_no_caching(self, monkeypatch):
        monkeypatch.setenv("SWARMS_API_KEY", "first-key")
        assert AgentMarketplaceHandler.check_api_key() == "first-key"

        monkeypatch.setenv("SWARMS_API_KEY", "second-key")
        assert AgentMarketplaceHandler.check_api_key() == "second-key"


########################################################
# _headers()
########################################################


class TestHeaders:
    def test_bearer_and_content_type(self, api_key):
        headers = AgentMarketplaceHandler._headers()
        assert headers == {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    def test_raises_without_key(self, no_api_key):
        with pytest.raises(ValueError):
            AgentMarketplaceHandler._headers()


########################################################
# fetch()
########################################################


class TestFetch:
    def test_by_prompt_id(self, api_key, monkeypatch):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return json_response(
                200,
                {
                    "name": "N",
                    "description": "D",
                    "prompt": "P",
                },
            )

        install_get_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.fetch(prompt_id="abc-123")

        assert result == ("N", "D", "P")
        assert (
            seen["url"]
            == f"{MARKETPLACE_BASE_URL}/get-prompts/abc-123"
        )

    def test_by_name_url_encodes_spaces(self, api_key, monkeypatch):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return json_response(200, {"name": "n"})

        install_get_transport(monkeypatch, handler)
        AgentMarketplaceHandler.fetch(name="code review assistant")

        assert "%20" in seen["url"]
        assert " " not in seen["url"]

    def test_by_name_url_encodes_slash_and_hash(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return json_response(200, {"name": "n"})

        install_get_transport(monkeypatch, handler)
        AgentMarketplaceHandler.fetch(name="a/b#c")

        assert "%2F" in seen["url"]
        assert "%23" in seen["url"]
        assert "a/b#c" not in seen["url"]

    def test_prompt_id_takes_precedence_over_name(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return json_response(200, {"name": "n"})

        install_get_transport(monkeypatch, handler)
        AgentMarketplaceHandler.fetch(
            prompt_id="the-id", name="the-name"
        )

        assert seen["url"].endswith("/get-prompts/the-id")
        assert "the-name" not in seen["url"]

    def test_return_params_on_true_returns_tuple(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(
                200,
                {
                    "name": "N",
                    "description": "D",
                    "prompt": "P",
                    "extra_field": "ignored",
                },
            )

        install_get_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.fetch(
            prompt_id="x", return_params_on=True
        )

        assert result == ("N", "D", "P")

    def test_return_params_on_false_returns_full_dict(
        self, api_key, monkeypatch
    ):
        payload = {
            "name": "N",
            "description": "D",
            "prompt": "P",
            "extra_field": "kept",
        }

        def handler(request):
            return json_response(200, payload)

        install_get_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.fetch(
            prompt_id="x", return_params_on=False
        )

        assert result == payload

    def test_404_returns_none(self, api_key, monkeypatch):
        def handler(request):
            return httpx.Response(404)

        install_get_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.fetch(prompt_id="missing")

        assert result is None

    def test_500_raises_http_status_error(self, api_key, monkeypatch):
        def handler(request):
            return httpx.Response(500, text="server exploded")

        install_get_transport(monkeypatch, handler)
        with pytest.raises(httpx.HTTPStatusError):
            AgentMarketplaceHandler.fetch(prompt_id="x")

    def test_neither_argument_raises_value_error(self, api_key):
        with pytest.raises(ValueError, match="prompt_id or name"):
            AgentMarketplaceHandler.fetch()

    def test_missing_json_keys_yield_none_entries(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(200, {"name": "OnlyName"})

        install_get_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.fetch(prompt_id="x")

        assert result == ("OnlyName", None, None)

    def test_request_carries_auth_header(self, api_key, monkeypatch):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            seen["content_type"] = request.headers.get("content-type")
            return json_response(200, {})

        install_get_transport(monkeypatch, handler)
        AgentMarketplaceHandler.fetch(prompt_id="x")

        assert seen["auth"] == f"Bearer {API_KEY}"
        assert seen["content_type"] == "application/json"

    def test_timeout_argument_is_honored(self, api_key, monkeypatch):
        def handler(request):
            return json_response(200, {})

        calls = install_get_transport(monkeypatch, handler)
        AgentMarketplaceHandler.fetch(prompt_id="x", timeout=7.5)

        assert calls == [7.5]

    def test_fetch_without_api_key_raises(
        self, no_api_key, monkeypatch
    ):
        # _headers() runs before the request is sent, so no transport is
        # needed here — but install one anyway to prove it's never hit.
        def handler(request):
            raise AssertionError(
                "request should not be sent without an API key"
            )

        install_get_transport(monkeypatch, handler)
        with pytest.raises(ValueError):
            AgentMarketplaceHandler.fetch(prompt_id="x")


########################################################
# fetch_prompt()
########################################################


class TestFetchPrompt:
    def test_returns_tuple(self, api_key, monkeypatch):
        def handler(request):
            return json_response(
                200,
                {"name": "N", "description": "D", "prompt": "P"},
            )

        install_get_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.fetch_prompt("some-id")

        assert result == ("N", "D", "P")

    def test_raises_value_error_when_not_found(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return httpx.Response(404)

        install_get_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="some-missing-id"):
            AgentMarketplaceHandler.fetch_prompt("some-missing-id")


########################################################
# add_prompt()
########################################################


class TestAddPrompt:
    def test_posts_correct_payload(self, api_key, monkeypatch):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            seen["auth"] = request.headers.get("authorization")
            return json_response(200, {"id": "new-prompt-id"})

        install_post_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.add_prompt(
            name="My Prompt",
            prompt="Do the thing.",
            description="A prompt that does the thing.",
            use_cases=[{"title": "T", "description": "D"}],
            tags="a,b",
            is_free=False,
            price_usd=4.99,
            category="coding",
        )

        assert seen["url"] == f"{MARKETPLACE_BASE_URL}/add-prompt"
        assert seen["auth"] == f"Bearer {API_KEY}"
        assert seen["body"] == {
            "name": "My Prompt",
            "prompt": "Do the thing.",
            "description": "A prompt that does the thing.",
            "useCases": [{"title": "T", "description": "D"}],
            "tags": "a,b",
            "is_free": False,
            "price_usd": 4.99,
            "category": "coding",
        }
        # "use_cases" (snake_case) must NOT appear in the payload.
        assert "use_cases" not in seen["body"]
        assert result == {"id": "new-prompt-id"}

    def test_tags_none_becomes_empty_string(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return json_response(200, {})

        install_post_transport(monkeypatch, handler)
        AgentMarketplaceHandler.add_prompt(
            name="N",
            prompt="P",
            description="D",
            use_cases=[{"title": "T", "description": "D"}],
            tags=None,
        )

        assert seen["body"]["tags"] == ""

    def test_is_free_price_category_pass_through(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return json_response(200, {})

        install_post_transport(monkeypatch, handler)
        AgentMarketplaceHandler.add_prompt(
            name="N",
            prompt="P",
            description="D",
            use_cases=[{"title": "T", "description": "D"}],
            is_free=False,
            price_usd=12.5,
            category="business",
        )

        assert seen["body"]["is_free"] is False
        assert seen["body"]["price_usd"] == 12.5
        assert seen["body"]["category"] == "business"

    def test_default_category(self, api_key, monkeypatch):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return json_response(200, {})

        install_post_transport(monkeypatch, handler)
        AgentMarketplaceHandler.add_prompt(
            name="N",
            prompt="P",
            description="D",
            use_cases=[{"title": "T", "description": "D"}],
        )

        assert seen["body"]["category"] == DEFAULT_CATEGORY

    @pytest.mark.parametrize(
        "missing_field",
        ["name", "prompt", "description", "category", "use_cases"],
    )
    def test_missing_required_field_raises_value_error(
        self, api_key, missing_field
    ):
        kwargs = dict(
            name="N",
            prompt="P",
            description="D",
            category="research",
            use_cases=[{"title": "T", "description": "D"}],
        )
        kwargs[missing_field] = None

        with pytest.raises(ValueError, match=missing_field):
            AgentMarketplaceHandler.add_prompt(**kwargs)

    def test_4xx_raises(self, api_key, monkeypatch):
        def handler(request):
            return json_response(400, {"error": "bad request"})

        install_post_transport(monkeypatch, handler)
        with pytest.raises(httpx.HTTPStatusError):
            AgentMarketplaceHandler.add_prompt(
                name="N",
                prompt="P",
                description="D",
                use_cases=[{"title": "T", "description": "D"}],
            )

    def test_5xx_raises(self, api_key, monkeypatch):
        def handler(request):
            return json_response(500, {"error": "boom"})

        install_post_transport(monkeypatch, handler)
        with pytest.raises(httpx.HTTPStatusError):
            AgentMarketplaceHandler.add_prompt(
                name="N",
                prompt="P",
                description="D",
                use_cases=[{"title": "T", "description": "D"}],
            )

    def test_non_json_response_returned_as_text(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return httpx.Response(200, text="not json at all")

        install_post_transport(monkeypatch, handler)
        result = AgentMarketplaceHandler.add_prompt(
            name="N",
            prompt="P",
            description="D",
            use_cases=[{"title": "T", "description": "D"}],
        )

        assert result == "not json at all"

    def test_add_prompt_without_api_key_raises(self, no_api_key):
        with pytest.raises(ValueError):
            AgentMarketplaceHandler.add_prompt(
                name="N",
                prompt="P",
                description="D",
                use_cases=[{"title": "T", "description": "D"}],
            )


########################################################
# load_prompt()
########################################################


def make_offline_agent(**overrides):
    kwargs = dict(
        agent_name=DEFAULT_AGENT_NAME,
        model_name="gpt-4o-mini",
        persistent_memory=False,
        print_on=False,
    )
    kwargs.update(overrides)
    return Agent(**kwargs)


class TestLoadPrompt:
    def test_appends_prompt_to_system_prompt(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(
                200,
                {
                    "name": "N",
                    "description": "D",
                    "prompt": "MARKETPLACE-PROMPT-BODY",
                },
            )

        install_get_transport(monkeypatch, handler)
        agent = make_offline_agent()
        before = agent.system_prompt

        agent.marketplace.load_prompt("some-id")

        assert (
            agent.system_prompt == before + "MARKETPLACE-PROMPT-BODY"
        )

    def test_backfills_name_only_when_default(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(
                200,
                {
                    "name": "Marketplace Name",
                    "description": None,
                    "prompt": "",
                },
            )

        install_get_transport(monkeypatch, handler)
        agent = make_offline_agent(agent_name=DEFAULT_AGENT_NAME)

        agent.marketplace.load_prompt("some-id")

        assert agent.agent_name == "Marketplace Name"
        assert agent.name == "Marketplace Name"

    def test_backfills_description_only_when_none(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(
                200,
                {
                    "name": None,
                    "description": "Marketplace Description",
                    "prompt": "",
                },
            )

        install_get_transport(monkeypatch, handler)
        agent = make_offline_agent(agent_description=None)

        agent.marketplace.load_prompt("some-id")

        assert agent.agent_description == "Marketplace Description"
        assert agent.description == "Marketplace Description"

    def test_leaves_explicitly_named_agent_untouched(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(
                200,
                {
                    "name": "Marketplace Name",
                    "description": "Marketplace Description",
                    "prompt": "",
                },
            )

        install_get_transport(monkeypatch, handler)
        agent = make_offline_agent(
            agent_name="MyCustomAgent",
            agent_description="My custom description",
        )

        agent.marketplace.load_prompt("some-id")

        assert agent.agent_name == "MyCustomAgent"
        assert agent.agent_description == "My custom description"

    def test_falls_back_to_agents_marketplace_prompt_id(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return json_response(
                200, {"name": "N", "description": "D", "prompt": "P"}
            )

        install_get_transport(monkeypatch, handler)
        agent = make_offline_agent()
        agent.marketplace_prompt_id = "configured-fallback-id"

        agent.marketplace.load_prompt()

        assert seen["url"].endswith(
            "/get-prompts/configured-fallback-id"
        )

    def test_reraises_when_prompt_missing(self, api_key, monkeypatch):
        def handler(request):
            return httpx.Response(404)

        install_get_transport(monkeypatch, handler)
        agent = make_offline_agent()

        with pytest.raises(ValueError, match="not found"):
            agent.marketplace.load_prompt("missing-id")


########################################################
# build_tags()
########################################################


class TestBuildTags:
    def test_tags_and_capabilities_joined(self):
        agent = make_offline_agent(
            tags=["a", "b"], capabilities=["c"]
        )
        assert agent.marketplace.build_tags() == "a, b, c"

    def test_tags_only(self):
        agent = make_offline_agent(tags=["a", "b"])
        assert agent.marketplace.build_tags() == "a, b"

    def test_capabilities_only(self):
        agent = make_offline_agent(capabilities=["x", "y"])
        assert agent.marketplace.build_tags() == "x, y"

    def test_neither_returns_empty_string_and_does_not_raise(self):
        """Regression guard: build_tags() used to raise (or otherwise
        misbehave) when both agent.tags and agent.capabilities were None.
        It must return "" cleanly."""
        agent = make_offline_agent(tags=None, capabilities=None)
        assert agent.marketplace.build_tags() == ""


########################################################
# publish()
########################################################


class TestPublish:
    def test_raises_agent_initialization_error_without_use_cases(
        self,
    ):
        agent = make_offline_agent(use_cases=None)

        with pytest.raises(AgentInitializationError):
            agent.marketplace.publish()

    def test_posts_name_conversation_description_and_merged_tags(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return json_response(200, {"id": "published"})

        install_post_transport(monkeypatch, handler)
        agent = make_offline_agent(
            agent_name="PublishedAgent",
            agent_description="A published agent",
            use_cases=[{"title": "T", "description": "D"}],
            tags=["tag1"],
            capabilities=["cap1"],
        )

        result = agent.marketplace.publish(category="content")

        body = seen["body"]
        assert body["name"] == "PublishedAgent"
        assert body["description"] == "A published agent"
        assert body["prompt"] == agent.short_memory.get_str()
        assert body["useCases"] == [
            {"title": "T", "description": "D"}
        ]
        assert body["tags"] == "tag1, cap1"
        assert body["category"] == "content"
        assert result == {"id": "published"}

    def test_default_category_used_when_unspecified(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return json_response(200, {})

        install_post_transport(monkeypatch, handler)
        agent = make_offline_agent(
            use_cases=[{"title": "T", "description": "D"}],
        )

        agent.marketplace.publish()

        assert seen["body"]["category"] == DEFAULT_CATEGORY


########################################################
# Standalone use: no agent required
########################################################


class TestStandaloneHandler:
    def test_constructs_with_no_agent(self):
        handler = AgentMarketplaceHandler()
        assert handler.agent is None

    def test_classmethods_usable_without_agent(
        self, api_key, monkeypatch
    ):
        def get_handler(request):
            return json_response(
                200, {"name": "N", "description": "D", "prompt": "P"}
            )

        install_get_transport(monkeypatch, get_handler)
        handler = AgentMarketplaceHandler()

        assert handler.fetch(prompt_id="x") == ("N", "D", "P")
        assert handler.fetch_prompt("x") == ("N", "D", "P")

    def test_add_prompt_usable_without_agent(
        self, api_key, monkeypatch
    ):
        def post_handler(request):
            return json_response(200, {"ok": True})

        install_post_transport(monkeypatch, post_handler)
        handler = AgentMarketplaceHandler()

        result = handler.add_prompt(
            name="N",
            prompt="P",
            description="D",
            use_cases=[{"title": "T", "description": "D"}],
        )
        assert result == {"ok": True}


########################################################
# Agent integration
########################################################


class TestAgentIntegration:
    def test_marketplace_prompt_id_loads_at_construction(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return json_response(
                200,
                {
                    "name": "ConstructedName",
                    "description": "ConstructedDescription",
                    "prompt": "CONSTRUCTED-PROMPT",
                },
            )

        install_get_transport(monkeypatch, handler)
        agent = Agent(
            model_name="gpt-4o-mini",
            persistent_memory=False,
            print_on=False,
            marketplace_prompt_id="ctor-prompt-id",
        )

        assert seen["url"].endswith("/get-prompts/ctor-prompt-id")
        assert agent.agent_name == "ConstructedName"
        assert agent.system_prompt.endswith("CONSTRUCTED-PROMPT")

    def test_publish_to_marketplace_true_publishes_at_construction(
        self, api_key, monkeypatch
    ):
        seen = {}

        def handler(request):
            seen["called"] = True
            seen["body"] = json.loads(request.content)
            return json_response(200, {"id": "auto-published"})

        install_post_transport(monkeypatch, handler)
        agent = Agent(
            agent_name="AutoPublishAgent",
            model_name="gpt-4o-mini",
            persistent_memory=False,
            print_on=False,
            publish_to_marketplace=True,
            use_cases=[{"title": "T", "description": "D"}],
        )

        assert seen.get("called") is True
        assert seen["body"]["name"] == "AutoPublishAgent"
        assert agent.publish_to_marketplace is True

    def test_handle_publish_to_marketplace_delegates(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(200, {"id": "delegated"})

        install_post_transport(monkeypatch, handler)
        agent = make_offline_agent(
            use_cases=[{"title": "T", "description": "D"}]
        )

        result = agent.handle_publish_to_marketplace()

        assert result == {"id": "delegated"}

    def test_load_prompt_from_marketplace_delegates(
        self, api_key, monkeypatch
    ):
        def handler(request):
            return json_response(
                200,
                {
                    "name": "DelegatedName",
                    "description": None,
                    "prompt": "DELEGATED-PROMPT",
                },
            )

        install_get_transport(monkeypatch, handler)
        agent = make_offline_agent()
        agent.marketplace_prompt_id = "delegated-id"

        agent._load_prompt_from_marketplace()

        assert agent.agent_name == "DelegatedName"
        assert agent.system_prompt.endswith("DELEGATED-PROMPT")


########################################################
# Error hierarchy: swarms/schemas/agent_errors.py
########################################################


ERROR_CLASS_NAMES = [
    "AgentInitializationError",
    "AgentRunError",
    "AgentLLMError",
    "AgentToolError",
    "AgentMemoryError",
    "AgentLLMInitializationError",
    "AgentToolExecutionError",
]


class TestAgentErrorHierarchy:
    def test_base_class_subclasses_exception(self):
        assert issubclass(schema_errors.AgentError, Exception)

    @pytest.mark.parametrize("class_name", ERROR_CLASS_NAMES)
    def test_each_error_subclasses_agent_error(self, class_name):
        cls = getattr(schema_errors, class_name)
        assert issubclass(cls, schema_errors.AgentError)

    @pytest.mark.parametrize("class_name", ERROR_CLASS_NAMES)
    def test_each_error_catchable_as_agent_error(self, class_name):
        cls = getattr(schema_errors, class_name)
        with pytest.raises(schema_errors.AgentError):
            raise cls("boom")

    @pytest.mark.parametrize(
        "class_name", ["AgentError"] + ERROR_CLASS_NAMES
    )
    def test_importable_from_schemas_and_structs_agent_same_object(
        self, class_name
    ):
        from_schemas = getattr(schema_errors, class_name)
        from_structs = getattr(agent_module, class_name)
        assert from_schemas is from_structs

    def test_agent_initialization_error_used_by_publish(self):
        # Sanity check that the handler actually raises the schema class,
        # not a look-alike defined elsewhere.
        agent = make_offline_agent(use_cases=None)
        try:
            agent.marketplace.publish()
        except schema_errors.AgentInitializationError:
            pass
        else:
            pytest.fail(
                "expected AgentInitializationError to be raised"
            )
