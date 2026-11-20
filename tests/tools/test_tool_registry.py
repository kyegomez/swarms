import json
import time

from swarms.tools import ToolStorage


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_tool_storage_stamps_the_registry_and_each_tool_when_created(
    monkeypatch,
):
    """The registry and each tool record the time they were made, not the import time."""
    real_gmtime = time.gmtime
    clock = [1893553445]
    monkeypatch.setattr(
        time, "gmtime", lambda *args: real_gmtime(clock[0])
    )

    storage = ToolStorage(name="math", description="Math tools")
    clock[0] += 60
    storage.add_tool(add)

    listed = json.loads(storage.list_tools())

    assert listed["time_created"] == "2030-01-02 03:04:05"
    assert listed["tools"][0]["time_created"] == "2030-01-02 03:05:05"
