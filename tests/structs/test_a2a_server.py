from starlette.testclient import TestClient

from swarms.structs.a2a_server import A2AServer

client = TestClient(A2AServer(str.upper, api_keys=["k"]).app)
message = {"role": "ROLE_USER", "parts": [{"text": "hi"}]}
body = {"jsonrpc": "2.0", "id": 7, "method": "SendMessage"}
body["params"] = {"message": {**message, "messageId": "m"}}


def test_public_card_and_send_message_round_trip():
    card = client.get("/.well-known/agent-card.json").json()
    url = card["supportedInterfaces"][0]["url"]
    assert client.post(url, json=body).status_code == 401
    headers = {"a2a-version": "1.0", "x-api-key": "k"}
    reply = client.post(url, json=body, headers=headers).json()
    assert reply["result"]["message"]["parts"] == [{"text": "HI"}]


def test_missing_version_header_is_version_not_supported():
    reply = client.post("/a2a", json=body, headers={"x-api-key": "k"})
    assert reply.json()["error"]["code"] == -32009


def test_init_options_override_the_defaults():
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
    reply = custom.post("/rpc", json=body, headers=headers).json()
    assert reply["result"]["message"]["parts"] == [{"text": "HI"}]
