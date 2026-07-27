"""
Test suite for :class:`swarms.tools.mcp_manager.MCPManager`.

Everything here runs against real MCP servers — no mocks:

* ``mcp_test_server.py`` is launched as a subprocess and speaks the real MCP
  protocol over streamable HTTP. Three instances are started: one open, one
  requiring an API key header, one requiring a bearer token.
* A handful of tests hit DeepWiki's public remote MCP server
  (https://mcp.deepwiki.com/mcp). They are marked ``remote`` and skip
  automatically when the network is unavailable. Skip them explicitly with
  ``pytest -m "not remote"``.

Run:
    pytest tests/tools/test_mcp_manager.py -v
    pytest tests/tools/test_mcp_manager.py -v -m "not remote"
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from swarms.schemas.agent_mcp_errors import AgentMCPConnectionError
from swarms.schemas.mcp_schemas import MCPConnection, MCPOAuthConfig
from swarms.tools.mcp_manager import (
    MCPFileTokenStorage,
    MCPInMemoryTokenStorage,
    MCPManager,
    _describe_exception,
    _OAuthCallbackServer,
    _resolve_secret,
    run_async,
)

SERVER_SCRIPT = Path(__file__).parent / "mcp_test_server.py"
API_KEY = "test-key-123"
BEARER_TOKEN = "test-token-abc"
REMOTE_URL = "https://mcp.deepwiki.com/mcp"


########################################################
# Fixtures: real MCP servers
########################################################


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(
                ("127.0.0.1", port), timeout=0.5
            ):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(
        f"MCP test server on port {port} never started"
    )


class _Server:
    def __init__(self, profile: str):
        self.port = _free_port()
        self.profile = profile
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(SERVER_SCRIPT),
                str(self.port),
                profile,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_port(self.port)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


@pytest.fixture(scope="session")
def open_server():
    server = _Server("open")
    yield server
    server.stop()


@pytest.fixture(scope="session")
def apikey_server():
    server = _Server("apikey")
    yield server
    server.stop()


@pytest.fixture(scope="session")
def bearer_server():
    server = _Server("bearer")
    yield server
    server.stop()


@pytest.fixture(scope="session")
def second_open_server():
    """A second open server, for multi-server routing tests."""
    server = _Server("open")
    yield server
    server.stop()


@pytest.fixture
def open_manager(open_server):
    return MCPManager(
        mcp_url=open_server.url,
        agent_name="test-agent",
        retry_attempts=1,
    )


########################################################
# Configuration normalization
########################################################


class TestConnectionNormalization:
    def test_url_string(self):
        manager = MCPManager(mcp_url="http://localhost:9/mcp")
        assert manager.enabled is True
        assert len(manager) == 1
        assert manager.connections[0].url == "http://localhost:9/mcp"

    def test_dict_and_connection_objects(self):
        manager = MCPManager(
            mcp_config={"url": "http://localhost:1/mcp"},
            mcp_configs=[
                MCPConnection(
                    url="http://localhost:2/mcp", name="two"
                )
            ],
        )
        assert len(manager) == 2
        assert manager.label(manager.connections[1]) == "two"

    def test_many_urls_and_dedupe(self):
        manager = MCPManager(
            mcp_url="http://localhost:1/mcp",
            mcp_urls=[
                "http://localhost:1/mcp",  # duplicate, dropped
                "http://localhost:2/mcp",
            ],
        )
        assert len(manager) == 2

    def test_no_servers_is_disabled(self):
        manager = MCPManager()
        assert manager.enabled is False
        assert bool(manager) is False
        assert manager.get_tools() == []

    def test_manager_defaults_applied_to_connections(self):
        manager = MCPManager(
            mcp_url="http://localhost:1/mcp",
            api_key="key-1",
            headers={"X-Trace": "abc"},
            transport="sse",
            timeout=99,
        )
        connection = manager.connections[0]
        assert connection.api_key == "key-1"
        assert connection.headers == {"X-Trace": "abc"}
        assert connection.timeout == 99
        assert manager._resolve_transport(connection) == "sse"

    def test_per_connection_auth_wins_over_default(self):
        manager = MCPManager(
            mcp_config=MCPConnection(
                url="http://localhost:1/mcp", api_key="specific"
            ),
            api_key="default",
        )
        assert manager.connections[0].api_key == "specific"

    def test_invalid_server_type_rejected(self):
        with pytest.raises(AgentMCPConnectionError):
            MCPManager(mcp_url=12345)

    def test_to_dict_redacts_secrets(self):
        manager = MCPManager(
            mcp_config=MCPConnection(
                url="http://localhost:1/mcp",
                api_key="super-secret",
                authorization_token="also-secret",
            )
        )
        blob = json.dumps(manager.to_dict())
        assert "super-secret" not in blob
        assert "also-secret" not in blob
        assert (
            manager.to_dict()["servers"][0]["auth_type"] == "api_key"
        )

    def test_add_server_clears_cache(self):
        manager = MCPManager(mcp_url="http://localhost:1/mcp")
        manager._tools_cache = [{"cached": True}]
        manager.add_server("http://localhost:2/mcp")
        assert len(manager) == 2
        assert manager._tools_cache is None


########################################################
# Authentication
########################################################


class TestAuthResolution:
    def headers_for(self, connection):
        manager = MCPManager(mcp_config=connection)
        return run_async(manager._build_headers(connection))

    def test_api_key_defaults_to_bearer_authorization(self):
        connection = MCPConnection(
            url="http://x/mcp", api_key="abc123"
        )
        assert self.headers_for(connection) == {
            "Authorization": "Bearer abc123"
        }

    def test_api_key_custom_header_without_prefix(self):
        connection = MCPConnection(
            url="http://x/mcp",
            api_key="abc123",
            api_key_header="X-API-Key",
            api_key_prefix=None,
        )
        assert self.headers_for(connection) == {"X-API-Key": "abc123"}

    def test_authorization_token_becomes_bearer(self):
        connection = MCPConnection(
            url="http://x/mcp", authorization_token="tok"
        )
        assert self.headers_for(connection) == {
            "Authorization": "Bearer tok"
        }

    def test_custom_headers_preserved(self):
        connection = MCPConnection(
            url="http://x/mcp", headers={"X-Tenant": "acme"}
        )
        assert self.headers_for(connection)["X-Tenant"] == "acme"

    def test_static_oauth_access_token(self):
        connection = MCPConnection(
            url="http://x/mcp",
            oauth=MCPOAuthConfig(access_token="oauth-token"),
        )
        assert self.headers_for(connection) == {
            "Authorization": "Bearer oauth-token"
        }

    def test_env_indirection_both_syntaxes(self, monkeypatch):
        monkeypatch.setenv("SWARMS_TEST_MCP_KEY", "from-env")
        for value in (
            "env:SWARMS_TEST_MCP_KEY",
            "${SWARMS_TEST_MCP_KEY}",
        ):
            connection = MCPConnection(
                url="http://x/mcp", api_key=value
            )
            assert self.headers_for(connection) == {
                "Authorization": "Bearer from-env"
            }

    def test_missing_env_var_resolves_to_none(self, monkeypatch):
        monkeypatch.delenv("SWARMS_TEST_MISSING", raising=False)
        assert _resolve_secret("env:SWARMS_TEST_MISSING") is None
        assert _resolve_secret("literal") == "literal"
        assert _resolve_secret(None) is None

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, "none"),
            ({"api_key": "k"}, "api_key"),
            ({"authorization_token": "t"}, "bearer"),
            ({"headers": {"X": "y"}}, "custom"),
            ({"oauth": MCPOAuthConfig()}, "oauth"),
            ({"auth_type": "custom", "api_key": "k"}, "custom"),
        ],
    )
    def test_auth_type_detection(self, kwargs, expected):
        manager = MCPManager()
        connection = MCPConnection(url="http://x/mcp", **kwargs)
        assert manager._resolve_auth_type(connection) == expected

    def test_oauth_provider_not_built_for_static_token(self):
        manager = MCPManager()
        connection = MCPConnection(
            url="http://x/mcp",
            oauth=MCPOAuthConfig(access_token="tok"),
        )
        assert manager._build_oauth_provider(connection) is None

    def test_oauth_provider_built_for_authorization_code(self):
        manager = MCPManager()
        connection = MCPConnection(
            url="https://x/mcp",
            oauth=MCPOAuthConfig(scopes=["mcp:tools"]),
        )
        provider = manager._build_oauth_provider(connection)
        assert provider is not None
        # cached per connection
        assert manager._build_oauth_provider(connection) is provider

    def test_client_credentials_requires_token_endpoint(self):
        manager = MCPManager()
        connection = MCPConnection(
            url="http://127.0.0.1:1/mcp",
            oauth=MCPOAuthConfig(
                grant_type="client_credentials", client_id="cid"
            ),
            timeout=2,
        )
        # No discoverable metadata on a dead port -> actionable error.
        with pytest.raises(AgentMCPConnectionError):
            run_async(manager._build_headers(connection))


########################################################
# Transport resolution
########################################################


class TestTransportResolution:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            (
                {"url": "https://x/mcp", "transport": "auto"},
                "streamable_http",
            ),
            ({"url": "https://x/sse", "transport": "auto"}, "sse"),
            ({"url": "https://x/mcp", "transport": "sse"}, "sse"),
            (
                {
                    "url": "https://x/mcp",
                    "transport": "streamable-http",
                },
                "streamable_http",
            ),
            (
                {"url": None, "command": "python", "args": ["s.py"]},
                "stdio",
            ),
            (
                {"url": "python server.py", "transport": "auto"},
                "stdio",
            ),
        ],
    )
    def test_transport(self, kwargs, expected):
        manager = MCPManager()
        assert (
            manager._resolve_transport(MCPConnection(**kwargs))
            == expected
        )


########################################################
# Tool call normalization
########################################################


class TestToolCallNormalization:
    def test_single_dict_with_json_arguments(self):
        calls = MCPManager._normalize_tool_calls(
            {
                "function": {
                    "name": "add",
                    "arguments": '{"a": 1, "b": 2}',
                }
            }
        )
        assert calls == [
            {"name": "add", "arguments": {"a": 1, "b": 2}}
        ]

    def test_list_of_calls(self):
        calls = MCPManager._normalize_tool_calls(
            [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": {"x": 1}}},
            ]
        )
        assert [c["name"] for c in calls] == ["a", "b"]

    def test_assistant_message_with_tool_calls(self):
        calls = MCPManager._normalize_tool_calls(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 5}',
                        },
                    }
                ],
            }
        )
        assert calls == [{"name": "add", "arguments": {"a": 5}}]

    def test_json_string_input(self):
        calls = MCPManager._normalize_tool_calls(
            json.dumps(
                [{"function": {"name": "add", "arguments": "{}"}}]
            )
        )
        assert calls == [{"name": "add", "arguments": {}}]

    @pytest.mark.parametrize(
        "response", [None, "", "not json", [], {}, {"content": "hi"}]
    )
    def test_non_tool_responses_yield_nothing(self, response):
        assert MCPManager._normalize_tool_calls(response) == []

    def test_malformed_arguments_default_to_empty_dict(self):
        calls = MCPManager._normalize_tool_calls(
            {"function": {"name": "add", "arguments": "{not json"}}
        )
        assert calls == [{"name": "add", "arguments": {}}]

    def test_pydantic_style_object(self):
        class FakeFunction:
            def __init__(self):
                self.name = "add"
                self.arguments = '{"a": 1}'

        class FakeCall:
            def __init__(self):
                self.function = {
                    "name": "add",
                    "arguments": '{"a": 1}',
                }

        assert MCPManager._normalize_tool_calls(FakeCall()) == [
            {"name": "add", "arguments": {"a": 1}}
        ]


########################################################
# Tool discovery against a real server
########################################################


class TestToolDiscovery:
    def test_lists_tools_as_openai_schemas(self, open_manager):
        tools = open_manager.get_tools()
        names = {t["function"]["name"] for t in tools}
        assert {"add", "greet", "boom", "whoami"} <= names
        add = next(t for t in tools if t["function"]["name"] == "add")
        assert add["type"] == "function"
        assert add["function"]["description"]
        assert add["function"]["parameters"]["type"] == "object"

    def test_mcp_format_returns_raw_tools(self, open_manager):
        tools = open_manager.get_tools(
            format="mcp", force_refresh=True
        )
        assert {t.name for t in tools} >= {"add", "greet"}

    def test_results_are_cached_until_refreshed(self, open_manager):
        first = open_manager.get_tools()
        assert open_manager.get_tools() is first
        open_manager.clear_cache()
        assert open_manager.get_tools() is not first

    def test_tool_routes_populated(self, open_manager):
        open_manager.get_tools()
        assert set(open_manager.list_tool_names()) >= {"add", "greet"}

    def test_api_key_server_accepts_correct_key(self, apikey_server):
        manager = MCPManager(
            mcp_config=MCPConnection(
                url=apikey_server.url,
                api_key=API_KEY,
                api_key_header="X-API-Key",
                api_key_prefix=None,
            ),
            retry_attempts=1,
        )
        assert {
            t["function"]["name"] for t in manager.get_tools()
        } >= {"add"}

    def test_api_key_server_rejects_wrong_key(self, apikey_server):
        manager = MCPManager(
            mcp_config=MCPConnection(
                url=apikey_server.url,
                api_key="wrong",
                api_key_header="X-API-Key",
                api_key_prefix=None,
            ),
            retry_attempts=1,
        )
        with pytest.raises(AgentMCPConnectionError) as excinfo:
            manager.get_tools()
        assert "401" in str(excinfo.value)

    def test_bearer_server_accepts_authorization_token(
        self, bearer_server
    ):
        manager = MCPManager(
            mcp_config=MCPConnection(
                url=bearer_server.url,
                authorization_token=BEARER_TOKEN,
            ),
            retry_attempts=1,
        )
        assert manager.get_tools()

    def test_bearer_server_accepts_manager_level_api_key(
        self, bearer_server
    ):
        manager = MCPManager(
            mcp_url=bearer_server.url,
            api_key=BEARER_TOKEN,
            retry_attempts=1,
        )
        assert manager.get_tools()

    def test_env_backed_credentials_work_end_to_end(
        self, apikey_server, monkeypatch
    ):
        monkeypatch.setenv("SWARMS_IT_MCP_KEY", API_KEY)
        manager = MCPManager(
            mcp_config=MCPConnection(
                url=apikey_server.url,
                api_key="env:SWARMS_IT_MCP_KEY",
                api_key_header="X-API-Key",
                api_key_prefix=None,
            ),
            retry_attempts=1,
        )
        assert manager.get_tools()

    def test_unreachable_server_raises(self):
        manager = MCPManager(
            mcp_url=f"http://127.0.0.1:{_free_port()}/mcp",
            retry_attempts=1,
            timeout=3,
        )
        with pytest.raises(AgentMCPConnectionError):
            manager.get_tools()

    def test_partial_failure_still_returns_working_tools(
        self, open_server
    ):
        manager = MCPManager(
            mcp_urls=[
                open_server.url,
                f"http://127.0.0.1:{_free_port()}/mcp",
            ],
            retry_attempts=1,
            timeout=3,
        )
        tools = manager.get_tools()
        assert {t["function"]["name"] for t in tools} >= {"add"}


########################################################
# Tool execution against a real server
########################################################


class TestToolExecution:
    def test_single_call(self, open_manager):
        results = open_manager.execute_tool_calls(
            [
                {
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 2, "b": 40}',
                    }
                }
            ]
        )
        assert len(results) == 1
        assert results[0]["is_error"] is False
        assert results[0]["result"] == "42"
        assert results[0]["tool"] == "add"

    def test_multiple_calls_preserve_order(self, open_manager):
        results = open_manager.execute_tool_calls(
            [
                {
                    "function": {
                        "name": "greet",
                        "arguments": '{"name": "A"}',
                    }
                },
                {
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 1, "b": 1}',
                    }
                },
                {
                    "function": {
                        "name": "greet",
                        "arguments": '{"name": "B"}',
                    }
                },
            ]
        )
        assert [r["tool"] for r in results] == [
            "greet",
            "add",
            "greet",
        ]
        assert results[0]["result"] == "Hello, A!"
        assert results[2]["result"] == "Hello, B!"

    def test_assistant_message_form(self, open_manager):
        results = open_manager.execute_tool_calls(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 3, "b": 4}',
                        },
                    }
                ],
            }
        )
        assert results[0]["result"] == "7"

    def test_server_side_tool_error_is_captured(self, open_manager):
        results = open_manager.execute_tool_calls(
            [{"function": {"name": "boom", "arguments": "{}"}}]
        )
        assert results[0]["is_error"] is True

    def test_unknown_tool_on_multi_server_setup(
        self, open_server, second_open_server
    ):
        manager = MCPManager(
            mcp_urls=[open_server.url, second_open_server.url],
            retry_attempts=1,
        )
        results = manager.execute_tool_calls(
            [
                {
                    "function": {
                        "name": "does_not_exist",
                        "arguments": "{}",
                    }
                }
            ]
        )
        assert results[0]["is_error"] is True
        assert "does_not_exist" in results[0]["error"]

    def test_no_tool_calls_returns_empty(self, open_manager):
        assert open_manager.execute_tool_calls("just text") == []

    def test_execution_without_servers_raises(self):
        with pytest.raises(AgentMCPConnectionError):
            MCPManager().execute_tool_calls(
                [{"function": {"name": "add", "arguments": "{}"}}]
            )

    def test_call_tool_helper(self, open_manager):
        result = open_manager.call_tool("greet", {"name": "Swarms"})
        assert result["result"] == "Hello, Swarms!"
        assert result["is_error"] is False

    def test_output_type_json(self, open_manager):
        out = open_manager.execute_tool_calls(
            [
                {
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 1, "b": 2}',
                    }
                }
            ],
            output_type="json",
        )
        assert isinstance(out, str)
        assert json.loads(out)[0]["result"] == "3"

    def test_output_type_str(self, open_manager):
        out = open_manager.execute_tool_calls(
            [
                {
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 1, "b": 2}',
                    }
                }
            ],
            output_type="str",
        )
        assert isinstance(out, str)
        assert "add" in out and "3" in out

    def test_authenticated_execution(self, apikey_server):
        manager = MCPManager(
            mcp_config=MCPConnection(
                url=apikey_server.url,
                api_key=API_KEY,
                api_key_header="X-API-Key",
                api_key_prefix=None,
            ),
            retry_attempts=1,
        )
        results = manager.execute_tool_calls(
            [
                {
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 20, "b": 22}',
                    }
                }
            ]
        )
        assert results[0]["result"] == "42"

    def test_calls_routed_to_owning_server(
        self, open_server, second_open_server
    ):
        """Each server exposes a `whoami` returning its own port."""
        manager_a = MCPManager(
            mcp_url=open_server.url, retry_attempts=1
        )
        manager_b = MCPManager(
            mcp_url=second_open_server.url, retry_attempts=1
        )
        a = manager_a.call_tool("whoami")["result"]
        b = manager_b.call_tool("whoami")["result"]
        assert a == f"server-{open_server.port}"
        assert b == f"server-{second_open_server.port}"
        assert a != b

    def test_multi_server_tools_are_merged(
        self, open_server, second_open_server
    ):
        manager = MCPManager(
            mcp_urls=[open_server.url, second_open_server.url],
            retry_attempts=1,
        )
        tools = manager.get_tools()
        # Same tool names on both servers: duplicates are dropped, not doubled.
        names = [t["function"]["name"] for t in tools]
        assert len(names) == len(set(names))
        assert (
            manager.execute_tool_calls(
                [
                    {
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 1, "b": 1}',
                        }
                    }
                ]
            )[0]["result"]
            == "2"
        )


########################################################
# Async / event loop behaviour
########################################################


class TestAsyncBehaviour:
    def test_async_api(self, open_manager):
        async def scenario():
            tools = await open_manager.aget_tools(force_refresh=True)
            results = await open_manager.aexecute_tool_calls(
                [
                    {
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 8, "b": 1}',
                        }
                    }
                ]
            )
            single = await open_manager.acall_tool(
                "greet", {"name": "Z"}
            )
            return tools, results, single

        tools, results, single = asyncio.run(scenario())
        assert tools
        assert results[0]["result"] == "9"
        assert single["result"] == "Hello, Z!"

    def test_sync_api_callable_from_inside_running_loop(
        self, open_manager
    ):
        """The sync helpers must not explode when a loop is already running."""

        async def scenario():
            return open_manager.get_tools(force_refresh=True), (
                open_manager.execute_tool_calls(
                    [
                        {
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 6, "b": 6}',
                            }
                        }
                    ]
                )
            )

        tools, results = asyncio.run(scenario())
        assert tools
        assert results[0]["result"] == "12"

    def test_concurrent_calls_from_many_tasks(self, open_manager):
        async def scenario():
            return await asyncio.gather(
                *[
                    open_manager.acall_tool("add", {"a": i, "b": i})
                    for i in range(5)
                ]
            )

        results = asyncio.run(scenario())
        assert [r["result"] for r in results] == [
            "0",
            "2",
            "4",
            "6",
            "8",
        ]


########################################################
# OAuth plumbing
########################################################


class TestOAuthPlumbing:
    def test_file_token_storage_round_trip(self, tmp_path):
        from mcp.shared.auth import OAuthToken

        path = tmp_path / "tokens.json"
        storage = MCPFileTokenStorage("https://x/mcp", str(path))

        assert run_async(storage.get_tokens()) is None

        token = OAuthToken(
            access_token="abc", token_type="Bearer", expires_in=3600
        )
        run_async(storage.set_tokens(token))

        restored = run_async(storage.get_tokens())
        assert restored.access_token == "abc"
        assert path.exists()
        assert oct(path.stat().st_mode)[-3:] == "600"

        storage.clear()
        assert not path.exists()
        assert run_async(storage.get_tokens()) is None

    def test_in_memory_storage_round_trip(self):
        from mcp.shared.auth import OAuthToken

        storage = MCPInMemoryTokenStorage()
        assert run_async(storage.get_tokens()) is None
        run_async(
            storage.set_tokens(
                OAuthToken(access_token="t", token_type="Bearer")
            )
        )
        assert run_async(storage.get_tokens()).access_token == "t"
        storage.clear()
        assert run_async(storage.get_tokens()) is None

    def test_default_storage_path_is_per_server(self):
        a = MCPFileTokenStorage("https://alpha.example.com/mcp")
        b = MCPFileTokenStorage("https://beta.example.com/mcp")
        assert a.path != b.path
        assert a.path.suffix == ".json"

    def test_use_token_cache_false_uses_memory_storage(self):
        manager = MCPManager()
        connection = MCPConnection(
            url="https://x/mcp",
            oauth=MCPOAuthConfig(use_token_cache=False),
        )
        provider = manager._build_oauth_provider(connection)
        assert isinstance(
            provider.context.storage, MCPInMemoryTokenStorage
        )

    def test_callback_server_captures_code_and_state(self):
        import threading
        import urllib.request

        port = _free_port()
        server = _OAuthCallbackServer(
            f"http://127.0.0.1:{port}/callback"
        )
        server.start()

        def hit():
            time.sleep(0.2)
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/callback?code=the-code&state=the-state"
            ).read()

        threading.Thread(target=hit, daemon=True).start()
        code, state = server.wait(timeout=15)
        assert code == "the-code"
        assert state == "the-state"

    def test_callback_server_surfaces_provider_error(self):
        import threading
        import urllib.error
        import urllib.request

        port = _free_port()
        server = _OAuthCallbackServer(
            f"http://127.0.0.1:{port}/callback"
        )
        server.start()

        def hit():
            time.sleep(0.2)
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/callback?error=access_denied"
                ).read()
            except urllib.error.HTTPError:
                pass

        threading.Thread(target=hit, daemon=True).start()
        with pytest.raises(AgentMCPConnectionError) as excinfo:
            server.wait(timeout=15)
        assert "access_denied" in str(excinfo.value)

    def test_callback_server_times_out(self):
        port = _free_port()
        server = _OAuthCallbackServer(
            f"http://127.0.0.1:{port}/callback"
        )
        server.start()
        with pytest.raises(AgentMCPConnectionError):
            server.wait(timeout=0.5)


########################################################
# Error formatting
########################################################


class TestErrorFormatting:
    def test_flattens_exception_groups(self):
        # anyio wraps transport failures in exception groups whose str() is
        # empty; builtin ExceptionGroup only exists on Python 3.11+.
        import builtins

        group_type = getattr(builtins, "ExceptionGroup", None)
        if group_type is None:
            pytest.skip("ExceptionGroup requires Python 3.11+")

        group = group_type(
            "boom",
            [ValueError("bad request: 401"), RuntimeError("closed")],
        )
        message = _describe_exception(group)
        assert "ValueError: bad request: 401" in message
        assert "RuntimeError: closed" in message

    def test_plain_exception(self):
        assert (
            _describe_exception(ValueError("nope"))
            == "ValueError: nope"
        )

    def test_message_less_exception_keeps_type(self):
        assert _describe_exception(ValueError()) == "ValueError"

    def test_none(self):
        assert _describe_exception(None) == "unknown error"


########################################################
# Agent integration
########################################################


class TestAgentIntegration:
    def test_agent_builds_manager_from_mcp_url(self, open_server):
        from swarms import Agent

        agent = Agent(
            agent_name="MCPTestAgent",
            model_name="gpt-5.4",
            mcp_url=open_server.url,
            max_loops=1,
            print_on=False,
            llm=object(),  # skip real LLM initialization
        )
        assert agent.mcp_enabled is True
        assert isinstance(agent.mcp_manager, MCPManager)

        tools = agent.add_mcp_tools_to_memory()
        assert {t["function"]["name"] for t in tools} >= {
            "add",
            "greet",
        }

    def test_agent_without_mcp_is_disabled(self):
        from swarms import Agent

        agent = Agent(
            agent_name="NoMCPAgent",
            model_name="gpt-5.4",
            max_loops=1,
            print_on=False,
            llm=object(),
        )
        assert agent.mcp_enabled is False

    def test_agent_passes_credentials_through(self, apikey_server):
        from swarms import Agent
        from swarms.schemas.mcp_schemas import MCPConnection as Conn

        agent = Agent(
            agent_name="MCPAuthAgent",
            model_name="gpt-5.4",
            mcp_config=Conn(
                url=apikey_server.url,
                api_key=API_KEY,
                api_key_header="X-API-Key",
                api_key_prefix=None,
            ),
            max_loops=1,
            print_on=False,
            llm=object(),
        )
        assert agent.add_mcp_tools_to_memory()

    def test_agent_mcp_tool_handling_executes(self, open_server):
        from swarms import Agent

        agent = Agent(
            agent_name="MCPExecAgent",
            model_name="gpt-5.4",
            mcp_url=open_server.url,
            max_loops=1,
            print_on=False,
            tool_call_summary=False,
            llm=object(),
        )
        results = agent.mcp_manager.execute_tool_calls(
            {
                "function": {
                    "name": "add",
                    "arguments": '{"a": 21, "b": 21}',
                }
            }
        )
        assert results[0]["result"] == "42"


########################################################
# Real remote MCP server (network)
########################################################


def _remote_available() -> bool:
    if os.getenv("SWARMS_SKIP_REMOTE_MCP_TESTS"):
        return False
    try:
        import httpx

        httpx.get("https://mcp.deepwiki.com", timeout=8)
        return True
    except Exception:
        return False


remote = pytest.mark.skipif(
    not _remote_available(),
    reason="remote MCP server unreachable (or SWARMS_SKIP_REMOTE_MCP_TESTS set)",
)


@pytest.mark.remote
@remote
class TestRealRemoteServer:
    """Hits DeepWiki's public, hosted MCP server over the internet."""

    def test_discovers_remote_tools(self):
        manager = MCPManager(mcp_url=REMOTE_URL, retry_attempts=2)
        names = {t["function"]["name"] for t in manager.get_tools()}
        assert "read_wiki_structure" in names

    def test_executes_remote_tool(self):
        manager = MCPManager(mcp_url=REMOTE_URL, retry_attempts=2)
        results = manager.execute_tool_calls(
            [
                {
                    "function": {
                        "name": "read_wiki_structure",
                        "arguments": '{"repoName": "kyegomez/swarms"}',
                    }
                }
            ]
        )
        assert results[0]["is_error"] is False
        assert "swarms" in str(results[0]["result"]).lower()

    def test_https_transport_auto_detected(self):
        manager = MCPManager(mcp_url=REMOTE_URL)
        assert (
            manager._resolve_transport(manager.connections[0])
            == "streamable_http"
        )
