"""Exercise real HTTP request encoding without sending any bytes."""

import http.client
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/real_estate/realtor_agent.py"
)


@pytest.mark.parametrize(
    "location",
    [
        "Menlo Park, CA",
        "San José, CA",
        "King & Queen County, VA",
        "A+B #1, Test",
    ],
)
def test_location_is_one_encoded_query_value(
    location: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the complete location and the separate sort parameter."""
    swarms = ModuleType("swarms")
    swarms.Agent = Mock()
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = Mock()
    monkeypatch.setitem(sys.modules, "swarms", swarms)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setenv("RAPID_API_KEY", "offline-test-placeholder")
    connections = []

    def offline_connection(host: str) -> http.client.HTTPConnection:
        connection = http.client.HTTPConnection(host)
        connection.send = Mock()
        connection.getresponse = Mock(
            return_value=Mock(read=Mock(return_value=b'{"data": []}'))
        )
        connection.request = Mock(wraps=connection.request)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        http.client, "HTTPSConnection", offline_connection
    )
    example = runpy.run_path(str(EXAMPLE))

    result = example["get_realtor_data_from_one_source"](location)

    assert json.loads(result) == {"data": []}
    endpoint = connections[-1].request.call_args.args[1]
    parsed = urlsplit(endpoint)
    assert parsed.fragment == ""
    assert parse_qs(parsed.query) == {
        "location": [f"city:{location}"],
        "sortBy": ["best_match"],
    }
    connections[-1].send.assert_called_once()
