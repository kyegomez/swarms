from httpx import MockTransport, Response
from pytest import mark, raises
from swarms.structs.a2a_client import A2AAgent

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

    return A2AAgent("http://r", "k", transport=MockTransport(serve))


@mark.parametrize("result", [{"message": TEXT}, {"task": JOB}])
def test_run_returns_reply_text(result):
    assert _remote({"result": result}).run(task="6x7") == "42"


@mark.parametrize("reply", [ERROR, {"result": FAILED}])
def test_run_raises_on_error_or_failed_task(reply):
    with raises(RuntimeError, match="A2A"):
        _remote(reply).run(task="6x7")
