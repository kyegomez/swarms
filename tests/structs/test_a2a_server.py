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
