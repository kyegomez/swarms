"""Check address-length validation without contacting a Solana node."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

EXAMPLES = (
    Path(__file__).resolve().parents[2]
    / "examples/single_agent/capabilities/tools/solana_tool"
)
ALPHABET = (
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
)


def encode_address(raw: bytes) -> str:
    number = int.from_bytes(raw, "big")
    suffix = ""
    while number:
        number, digit = divmod(number, 58)
        suffix = ALPHABET[digit] + suffix
    return "1" * (len(raw) - len(raw.lstrip(bytes([0])))) + suffix


@pytest.fixture(params=["solana_tool.py", "solana_tool_test.py"])
def fetch(request: pytest.FixtureRequest) -> Any:
    path = EXAMPLES / request.param
    method = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "fetch_wallet_transactions"
    )
    response = SimpleNamespace(json=lambda: {"result": []})
    session = SimpleNamespace(post=Mock(return_value=response))
    calls = Mock(return_value={"result": []})
    namespace = {
        "json": json,
        "logger": Mock(),
        "time": Mock(),
        "random": Mock(),
        "RPC_ENDPOINTS": ["offline"],
        "SolanaAPIException": RuntimeError,
        "TransactionError": lambda **kw: SimpleNamespace(**kw),
        "create_http_session": Mock(return_value=session),
        "get_working_endpoint": Mock(return_value="offline"),
        "RPCEndpointManager": Mock(),
        "make_request": calls,
    }
    exec(  # noqa: S102 - Exercise source with offline RPC boundaries.
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(
        run=namespace[method.name],
        session=session,
        request=calls,
        namespace=namespace,
    )


@pytest.mark.parametrize(
    "raw",
    [bytes(32), bytes(31) + b"\xff", b"\x01" * 32, b"\xff" * 32],
)
def test_base58_encoded_32_byte_addresses_are_accepted(
    fetch: Any, raw: bytes
) -> None:
    address = encode_address(raw)
    result = json.loads(fetch.run(address))
    assert result["success"] is True
    assert result["transactions"] == []
    assert fetch.session.post.called or fetch.request.called
    call = fetch.session.post.call_args or fetch.request.call_args
    payload = (
        call.kwargs["json"]
        if fetch.session.post.called
        else call.args[1]
    )
    assert payload["params"][0] == address


@pytest.mark.parametrize("value", ["1" * 31, "1" * 45, "", None, 123])
def test_invalid_type_or_length_is_rejected_before_rpc(
    fetch: Any, value: Any
) -> None:
    assert json.loads(fetch.run(value))["success"] is False
    fetch.session.post.assert_not_called()
    fetch.request.assert_not_called()
