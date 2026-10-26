import os
import socket
from typing import Any, NoReturn

import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider credentials and reject network access."""
    # Remove provider credentials from this test process only.
    for name in tuple(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def reject_network(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError(
            "Unexpected network access in an offline test"
        )

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
