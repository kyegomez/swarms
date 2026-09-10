import asyncio
import socket

import httpx
import pytest
from starlette.datastructures import Headers

from swarms.structs.mcp_deployer import (
    AuthResult,
    MCPDeployer,
    credential_from_headers,
)


class EchoAgent:
    agent_name = "Echo-Agent"
    agent_description = "Echoes the task back."

    def __init__(self):
        self.calls = []

    def run(self, task, img=None, **kwargs):
        self.calls.append((task, img))
        return f"echo: {task}"


def headers(**kwargs) -> Headers:
    return Headers(
        {k.replace("_", "-"): v for k, v in kwargs.items()}
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_refuses_to_build_without_any_auth():
    with pytest.raises(ValueError, match="No auth configured"):
        MCPDeployer(EchoAgent())


def test_allow_anonymous_builds_without_keys():
    deployer = MCPDeployer(EchoAgent(), allow_anonymous=True)
    assert deployer.tool_name == "echo_agent"
    assert deployer.description == "Echoes the task back."


def test_plain_function_target_uses_its_name_and_docstring():
    def summarise(task: str) -> str:
        """Summarise the task."""
        return task.upper()

    deployer = MCPDeployer(summarise, api_keys=["k"])
    assert deployer.tool_name == "summarise"
    assert deployer.description == "Summarise the task."
    assert deployer._invoke("summarise", "hi") == "HI"


def test_rejects_non_callable_target():
    with pytest.raises(ValueError, match="callable"):
        MCPDeployer(object(), api_keys=["k"])


def test_api_key_env_is_read_and_deduplicated(monkeypatch):
    monkeypatch.setenv("MCP_KEYS_TEST", "a, b ,a,")
    deployer = MCPDeployer(
        EchoAgent(), api_keys=["b"], api_key_env="MCP_KEYS_TEST"
    )
    assert deployer.api_keys == ["b", "a"]


def test_list_of_targets_becomes_one_tool_each():
    def summarise(task: str) -> str:
        """Summarise."""
        return task

    deployer = MCPDeployer(
        [EchoAgent(), summarise], api_keys=["k"], show_banner=False
    )
    assert deployer.tool_names == ["echo_agent", "summarise"]
    assert deployer.name == "echo_agent"
    assert deployer.tools["summarise"].description == "Summarise."


def test_dict_of_targets_uses_keys_as_tool_names():
    deployer = MCPDeployer(
        {"research": EchoAgent(), "shout": lambda t: t.upper()},
        api_keys=["k"],
        name="Team",
    )
    assert deployer.tool_names == ["research", "shout"]
    assert deployer.name == "Team"
    assert deployer._invoke("shout", "hi") == "HI"
    assert deployer._invoke("research", "hi") == "echo: hi"


def test_add_tool_registers_before_start_only():
    deployer = MCPDeployer(
        EchoAgent(), api_keys=["k"], show_banner=False
    )
    served = deployer.add_tool(lambda t: t[::-1], name="reverse")
    assert served.name == "reverse"
    assert deployer.tool_names == ["echo_agent", "reverse"]
    assert deployer._invoke("reverse", "abc") == "cba"


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError, match="already registered"):
        MCPDeployer([EchoAgent(), EchoAgent()], api_keys=["k"])


def test_tool_name_and_description_need_a_single_target():
    with pytest.raises(ValueError, match="single target"):
        MCPDeployer([EchoAgent()], api_keys=["k"], tool_name="x")
    with pytest.raises(ValueError, match="single target"):
        MCPDeployer(
            {"a": EchoAgent()}, api_keys=["k"], description="d"
        )


def test_unservable_entry_in_list_is_rejected():
    with pytest.raises(ValueError, match="callable"):
        MCPDeployer([EchoAgent(), object()], api_keys=["k"])


# ----------------------------------------------------------------------
# Credential extraction and authenticate()
# ----------------------------------------------------------------------


def test_credential_prefers_api_key_header_then_bearer():
    assert credential_from_headers(headers(x_api_key="k1")) == "k1"
    assert (
        credential_from_headers(headers(x_api_key="Bearer k2"))
        == "k2"
    )
    assert (
        credential_from_headers(headers(authorization="Bearer k3"))
        == "k3"
    )
    assert (
        credential_from_headers(headers(authorization="Basic x"))
        is None
    )
    assert credential_from_headers(headers()) is None


def test_api_key_auth_admits_only_configured_keys():
    deployer = MCPDeployer(EchoAgent(), api_keys=["good"])

    ok = asyncio.run(deployer.authenticate(headers(x_api_key="good")))
    assert ok.method == "api_key"
    assert (
        asyncio.run(deployer.authenticate(headers(x_api_key="bad")))
        is None
    )
    assert asyncio.run(deployer.authenticate(headers())) is None


def test_custom_auth_callable_sync_and_async():
    def sync_auth(credential, hdrs):
        return credential == "s" and {
            "subject": "alice",
            "scopes": ["r"],
        }

    deployer = MCPDeployer(EchoAgent(), auth=sync_auth)
    result = asyncio.run(
        deployer.authenticate(headers(x_api_key="s"))
    )
    assert result.method == "custom"
    assert result.subject == "alice"
    assert result.scopes == ["r"]
    assert (
        asyncio.run(deployer.authenticate(headers(x_api_key="x")))
        is None
    )

    async def async_auth(credential, hdrs):
        return hdrs.get("x-tenant") == "acme"

    deployer = MCPDeployer(EchoAgent(), auth=async_auth)
    assert asyncio.run(
        deployer.authenticate(headers(x_tenant="acme"))
    )
    assert asyncio.run(deployer.authenticate(headers())) is None


def test_custom_auth_wins_over_api_keys():
    deployer = MCPDeployer(
        EchoAgent(), api_keys=["good"], auth=lambda c, h: False
    )
    assert (
        asyncio.run(deployer.authenticate(headers(x_api_key="good")))
        is None
    )


def test_token_verifier_checks_scopes_and_expiry():
    from mcp.server.auth.provider import AccessToken

    class Verifier:
        async def verify_token(self, token):
            if token == "admin":
                return AccessToken(
                    token=token,
                    client_id="c1",
                    scopes=["read", "write"],
                )
            if token == "reader":
                return AccessToken(
                    token=token, client_id="c2", scopes=["read"]
                )
            if token == "stale":
                return AccessToken(
                    token=token,
                    client_id="c3",
                    scopes=["write"],
                    expires_at=1,
                )
            return None

    deployer = MCPDeployer(
        EchoAgent(),
        token_verifier=Verifier(),
        required_scopes=["write"],
    )
    admitted = asyncio.run(
        deployer.authenticate(headers(authorization="Bearer admin"))
    )
    assert admitted.method == "token"
    assert admitted.subject == "c1"
    for token in ("reader", "stale", "unknown"):
        assert (
            asyncio.run(
                deployer.authenticate(
                    headers(authorization=f"Bearer {token}")
                )
            )
            is None
        )


def test_invoke_forwards_img_and_renders_non_string_output():
    agent = EchoAgent()
    deployer = MCPDeployer(agent, api_keys=["k"])
    deployer._invoke("echo_agent", "look", img="chart.png")
    assert agent.calls[-1] == ("look", "chart.png")

    deployer = MCPDeployer(lambda task: {"a": 1}, api_keys=["k"])
    assert deployer._invoke("lambda", "x") == '{\n  "a": 1\n}'


# ----------------------------------------------------------------------
# Live server over streamable HTTP
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def live():
    agent = EchoAgent()

    def shout(task: str) -> str:
        """Upper-case the task."""
        return task.upper()

    deployer = MCPDeployer(
        [agent, shout],
        api_keys=["secret-key"],
        port=free_port(),
        verbose=True,
        show_banner=False,
    )
    with deployer:
        yield deployer, agent


def test_health_is_public(live):
    deployer, _ = live
    response = httpx.get(
        f"http://{deployer.host}:{deployer.port}/health"
    )
    assert response.status_code == 200
    assert response.json()["tool"] == "echo_agent"
    assert response.json()["tools"] == ["echo_agent", "shout"]


def test_mcp_endpoint_rejects_missing_and_wrong_keys(live):
    deployer, _ = live
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    hdrs = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }
    anonymous = httpx.post(deployer.url, json=body, headers=hdrs)
    assert anonymous.status_code == 401
    assert anonymous.headers["www-authenticate"] == "Bearer"

    wrong = httpx.post(
        deployer.url, json=body, headers={**hdrs, "x-api-key": "nope"}
    )
    assert wrong.status_code == 401


def test_mcp_client_lists_and_calls_the_tool_with_bearer_key(live):
    from swarms.tools.mcp_manager import MCPManager

    deployer, agent = live
    manager = MCPManager(mcp_url=deployer.url, api_key="secret-key")

    tools = manager.get_tools()
    assert sorted(t["function"]["name"] for t in tools) == [
        "echo_agent",
        "shout",
    ]
    for tool in tools:
        params = tool["function"]["parameters"]["properties"]
        assert set(params) == {"task", "img"}

    result = manager.call_tool("echo_agent", {"task": "ping"})
    assert "echo: ping" in str(result)
    assert agent.calls[-1] == ("ping", None)

    shouted = manager.call_tool("shout", {"task": "quiet"})
    assert "QUIET" in str(shouted)


def test_mcp_client_can_use_the_x_api_key_header(live):
    from swarms.schemas.mcp_schemas import MCPConnection
    from swarms.tools.mcp_manager import MCPManager

    deployer, _ = live
    manager = MCPManager(
        mcp_url=MCPConnection(
            url=deployer.url,
            api_key="secret-key",
            api_key_header="x-api-key",
            api_key_prefix=None,
        )
    )
    assert sorted(manager.list_tool_names()) == [
        "echo_agent",
        "shout",
    ]


def test_mcp_client_with_bad_key_fails(live):
    from swarms.tools.mcp_manager import MCPManager

    deployer, _ = live
    manager = MCPManager(
        mcp_url=deployer.url, api_key="wrong", retry_attempts=1
    )
    with pytest.raises(Exception):
        manager.get_tools()


def test_banner_renders_for_every_auth_mode():
    from io import StringIO
    from rich.console import Console

    def render(deployer):
        buf = StringIO()
        deployer.print_banner(
            Console(file=buf, width=100, force_terminal=False)
        )
        return buf.getvalue()

    keyed = render(
        MCPDeployer(EchoAgent(), api_keys=["a", "b"], port=1)
    )
    assert "MCPDeployer" in keyed and "echo_agent" in keyed
    assert "2 api keys" in keyed and "http://127.0.0.1:1/mcp" in keyed
    assert "Swarms MCP" in keyed

    custom = render(MCPDeployer(EchoAgent(), auth=lambda c, h: True))
    assert "custom auth callable" in custom

    open_ = render(
        MCPDeployer(
            EchoAgent(), allow_anonymous=True, transport="stdio"
        )
    )
    assert "anonymous (open)" in open_ and "stdin/stdout" in open_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
