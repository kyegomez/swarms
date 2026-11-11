from httpx import MockTransport, Response
from pytest import mark, raises
from starlette.testclient import TestClient

from swarms.structs.a2a_client import A2AClient
from swarms.structs.a2a_server import A2AServer

V1 = {"protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
CARD = dict(name="R", supportedInterfaces=[{"url": "http://r", **V1}])
TEXT = {"parts": [{"text": "42"}]}
JOB = dict(status={"state": "TASK_STATE_COMPLETED"}, artifacts=[TEXT])
FAILED = {"task": {"status": {"state": "TASK_STATE_FAILED"}}}
ERROR = {"error": {"code": -32603, "message": "boom"}}


def _remote(reply):
    def serve(request):
        assert request.headers["A2A-Version"] == "1.0"
        assert request.headers["Authorization"] == "Bearer k"
        return Response(200, json=reply if request.content else CARD)

    return A2AClient("http://r", "k", transport=MockTransport(serve))


@mark.parametrize("result", [{"message": TEXT}, {"task": JOB}])
def test_client_run_returns_reply_text(result):
    assert _remote({"result": result}).run(task="6x7") == "42"


@mark.parametrize("reply", [ERROR, {"result": FAILED}])
def test_client_run_raises_on_error_or_failed_task(reply):
    with raises(RuntimeError, match="A2A"):
        _remote(reply).run(task="6x7")


def test_client_init_options_override_the_defaults():
    seen = []

    def serve(request):
        seen.append((request.url.path, dict(request.headers)))
        version = {**V1, "protocolVersion": "2.0"}
        card = {
            **CARD,
            "supportedInterfaces": [{"url": "http://r/x", **version}],
        }
        return Response(200, json=card)

    agent = A2AClient(
        "http://r",
        transport=MockTransport(serve),
        timeout=5.0,
        headers={"X-Tenant": "acme"},
        card_path="/card.json",
        version="2.0",
        agent_name="Local name",
        agent_description="Local description",
    )

    path, headers = seen[0]
    assert (path, headers["a2a-version"], headers["x-tenant"]) == (
        "/card.json",
        "2.0",
        "acme",
    )
    assert agent.client.timeout.read == 5.0
    assert (agent.agent_name, agent.agent_description, agent.url) == (
        "Local name",
        "Local description",
        "http://r/x",
    )


SERVER = TestClient(A2AServer(str.upper, api_keys=["k"]).app)
MESSAGE = {"role": "ROLE_USER", "parts": [{"text": "hi"}]}
BODY = {"jsonrpc": "2.0", "id": 7, "method": "SendMessage"}
BODY["params"] = {"message": {**MESSAGE, "messageId": "m"}}


def test_server_public_card_and_send_message_round_trip():
    card = SERVER.get("/.well-known/agent-card.json").json()
    url = card["supportedInterfaces"][0]["url"]
    assert SERVER.post(url, json=BODY).status_code == 401
    headers = {"a2a-version": "1.0", "x-api-key": "k"}
    reply = SERVER.post(url, json=BODY, headers=headers).json()
    assert reply["result"]["message"]["parts"] == [{"text": "HI"}]


def test_server_missing_version_header_is_version_not_supported():
    reply = SERVER.post("/a2a", json=BODY, headers={"x-api-key": "k"})
    assert reply.json()["error"]["code"] == -32009


def test_server_init_options_override_the_defaults():
    skill = {
        "id": "shout",
        "name": "Shout",
        "description": "Upper-cases text",
    }
    custom = TestClient(
        A2AServer(
            str.upper,
            api_keys=["k"],
            card_path="/card.json",
            rpc_path="/rpc",
            protocol_version="2.0",
            agent_version="3.1.4",
            capabilities={"streaming": False},
            input_modes=["text/markdown"],
            output_modes=["application/json"],
            skills=[skill],
        ).app
    )

    card = custom.get("/card.json").json()
    interface = card["supportedInterfaces"][0]
    assert interface["url"].endswith("/rpc")
    assert interface["protocolVersion"] == "2.0"
    assert (
        card["version"],
        card["capabilities"],
        card["defaultInputModes"],
        card["defaultOutputModes"],
        card["skills"],
    ) == (
        "3.1.4",
        {"streaming": False},
        ["text/markdown"],
        ["application/json"],
        [skill],
    )

    headers = {"a2a-version": "2.0", "x-api-key": "k"}
    reply = custom.post("/rpc", json=BODY, headers=headers).json()
    assert reply["result"]["message"]["parts"] == [{"text": "HI"}]
