"""Tests for the hardened run_bash_tool.

Covers the security fix: commands are executed as parsed argv with
``shell=False`` (shell metacharacters are inert), and dangerous patterns are
rejected both on the raw string and on the parsed token list.
"""

import pytest

from swarms.structs.autonomous_loop_utils import (
    _check_bash_argv,
    _check_bash_command,
    run_bash_tool,
)


class _StubMemory:
    def __init__(self):
        self.entries = []

    def add(self, role, content):
        self.entries.append((role, content))


class _StubAgent:
    def __init__(self):
        self.short_memory = _StubMemory()
        self.verbose = False


@pytest.fixture
def stub_agent():
    return _StubAgent()


class TestRawStringBlocklist:
    def test_classic_rm_rf_blocked(self):
        assert _check_bash_command("rm -rf /") is not None

    def test_split_flag_bypass_blocked(self):
        # "rm -r -f /" evades the ("rm", "-rf") substring check.
        assert _check_bash_command("rm -r -f /") is not None

    def test_long_flag_bypass_blocked(self):
        assert _check_bash_command("rm --recursive --force /") is not None

    def test_chmod_recursive_bypass_blocked(self):
        assert _check_bash_command("chmod -R 777 /") is not None

    def test_benign_command_allowed(self):
        assert _check_bash_command("echo hello") is None


class TestArgvBlocklist:
    def test_quoted_concat_bypass_blocked(self):
        # r""m -rf / hides "rm" from the raw-string check; the parsed argv is
        # ["rm", "-rf", "/"] and must be caught at token level.
        argv = ["rm", "-rf", "/"]
        assert _check_bash_argv(argv) is not None

    def test_backslash_escape_bypass_blocked(self):
        argv = ["rm", "-rf", "/"]
        assert _check_bash_argv(argv) is not None

    def test_split_flag_bypass_blocked_at_argv_level(self):
        argv = ["rm", "-r", "-f", "/"]
        assert _check_bash_argv(argv) is not None

    def test_chmod_bypass_blocked_at_argv_level(self):
        argv = ["chmod", "-R", "777", "/"]
        assert _check_bash_argv(argv) is not None

    def test_empty_argv_rejected(self):
        assert _check_bash_argv([]) is not None

    def test_nul_byte_rejected(self):
        assert _check_bash_argv(["echo", "\x00"]) is not None

    def test_benign_argv_allowed(self):
        assert _check_bash_argv(["echo", "hello"]) is None


class TestRunBashTool:
    def test_rm_rf_blocked(self, stub_agent):
        result = run_bash_tool(stub_agent, "rm -r -f /")
        assert result.startswith("Error:")
        assert "blocked" in result.lower() or "dangerous" in result.lower()

    def test_quoted_concat_blocked(self, stub_agent):
        result = run_bash_tool(stub_agent, 'r""m -rf /')
        assert result.startswith("Error:")

    def test_chmod_recursive_blocked(self, stub_agent):
        result = run_bash_tool(stub_agent, "chmod -R 777 /")
        assert result.startswith("Error:")

    def test_unparseable_command_rejected(self, stub_agent):
        result = run_bash_tool(stub_agent, 'echo "unclosed')
        assert result.startswith("Error:")

    def test_shell_metacharacters_are_inert(self, stub_agent):
        # With shell=False, ">/dev/sda" is a literal argv token, not a
        # redirection: it cannot write to the disk, it just fails to exec.
        result = run_bash_tool(stub_agent, ">/dev/sda")
        assert result.startswith("Error")

    def test_pipe_to_sh_is_inert(self, stub_agent):
        result = run_bash_tool(stub_agent, "|sh")
        assert result.startswith("Error")

    def test_benign_command_executes(self, stub_agent):
        result = run_bash_tool(stub_agent, "echo hello")
        assert "hello" in result
        assert "exited with code 0" in result

    def test_blocked_command_recorded_in_memory(self, stub_agent):
        run_bash_tool(stub_agent, "rm -rf /")
        assert any(
            "Blocked" in content for _, content in stub_agent.short_memory.entries
        )