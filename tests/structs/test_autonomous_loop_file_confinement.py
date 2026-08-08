"""The autonomous-loop file tools must stay inside the agent workspace.

The model picks `file_path` itself, so any prompt injection reaching the loop
(a fetched page, a file the agent was asked to summarise, a sub-agent's task
string) otherwise becomes arbitrary local file read, overwrite or deletion
under the host process's privileges.

No agent, no network: the workspace is a tmp_path and the agent is a mock that
only answers _get_agent_workspace_dir.
"""

import os
from unittest.mock import MagicMock

import pytest

from swarms.structs.autonomous_loop_utils import (
    create_file_tool,
    delete_file_tool,
    grep_tool,
    list_directory_tool,
    read_file_tool,
    update_file_tool,
)


@pytest.fixture
def workspace(tmp_path):
    """A workspace with a secret sitting just outside it."""
    ws = tmp_path / "agent_workspace" / "agents" / "worker-1"
    ws.mkdir(parents=True)
    (ws / "inside.txt").write_text("INSIDE-OK\n")
    (tmp_path / "outside_secret.txt").write_text("SECRET-OUTSIDE\n")

    agent = MagicMock()
    agent._get_agent_workspace_dir.return_value = str(ws)
    return agent, ws, tmp_path / "outside_secret.txt"


def test_every_file_tool_refuses_to_escape(workspace):
    agent, ws, secret = workspace
    relative_escape = os.path.relpath(secret, ws)

    # signatures differ (grep takes the pattern first), so call each one
    for label, call in (
        (
            "read_file relative",
            lambda: read_file_tool(agent, relative_escape),
        ),
        (
            "read_file absolute",
            lambda: read_file_tool(agent, str(secret)),
        ),
        ("grep", lambda: grep_tool(agent, "SECRET", relative_escape)),
        (
            "list_directory",
            lambda: list_directory_tool(agent, str(secret.parent)),
        ),
        ("delete_file", lambda: delete_file_tool(agent, str(secret))),
    ):
        result = str(call())
        assert (
            "outside the agent workspace" in result
        ), f"{label} did not refuse the escape: {result[:120]}"
        assert "SECRET-OUTSIDE" not in result

    # writes must not land outside either, by relative or absolute path
    create_file_tool(agent, relative_escape, "overwritten")
    update_file_tool(agent, str(secret), "overwritten")
    assert secret.read_text() == "SECRET-OUTSIDE\n"


def test_legitimate_workspace_access_still_works(workspace):
    agent, ws, _ = workspace

    assert "INSIDE-OK" in str(read_file_tool(agent, "inside.txt"))
    # an absolute path inside the workspace is still fine: the check is
    # containment, not "relative paths only"
    assert "INSIDE-OK" in str(
        read_file_tool(agent, str(ws / "inside.txt"))
    )
    assert "inside.txt" in str(list_directory_tool(agent, ""))
    assert "Error" not in str(create_file_tool(agent, "new.txt", "x"))
    assert (ws / "new.txt").exists()
