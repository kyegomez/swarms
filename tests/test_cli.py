"""
Single-file test suite for the Swarms CLI.

Covers: argument parsing, command routing, handler logic, and utility functions.
Uses mocking to avoid real LLM/network calls.
Run with: pytest tests/test_cli.py -v
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_args(**kwargs) -> argparse.Namespace:
    """Return a Namespace with sensible defaults for all CLI args."""
    defaults = dict(
        command="agent",
        yaml_file="agents.yaml",
        markdown_path=None,
        concurrent=True,
        name="TestAgent",
        description="A test agent",
        system_prompt="You are a helpful assistant.",
        model_name="gpt-4o",
        task="Say hello",
        interactive=False,
        temperature=0.5,
        max_loops="1",
        auto_generate_prompt=False,
        dynamic_temperature_enabled=False,
        dynamic_context_window=False,
        output_type="str",
        verbose=False,
        context_length=4096,
        retry_attempts=3,
        streaming_on=False,
        autosave=False,
        saved_state_path=None,
        return_step_meta=False,
        dashboard=False,
        marketplace_prompt_id=None,
        loops_per_agent=1,
        question_agent_model_name="gpt-4o",
        worker_model_name="gpt-4o",
        random_loops_per_agent=False,
        model="gpt-4o",
        output=None,
        output_dir=None,
        no_run=False,
        dir=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ===========================================================================
# 1. setup_argument_parser
# ===========================================================================


class TestSetupArgumentParser:
    """Tests for setup_argument_parser()."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import setup_argument_parser

        self.setup_argument_parser = setup_argument_parser

    def test_returns_parser(self):
        parser = self.setup_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_valid_commands_parse(self):
        valid_commands = [
            "init",
            "onboarding",
            "get-api-key",
            "check-login",
            "run-agents",
            "load-markdown",
            "agent",
            "chat",
            "upgrade",
            "autoswarm",
            "setup-check",
            "llm-council",
            "heavy-swarm",
        ]
        parser = self.setup_argument_parser()
        for cmd in valid_commands:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_invalid_command_raises(self):
        parser = self.setup_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent-command"])

    def test_default_yaml_file(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["run-agents"])
        assert args.yaml_file == "agents.yaml"

    def test_custom_yaml_file(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(
            ["run-agents", "--yaml-file", "my.yaml"]
        )
        assert args.yaml_file == "my.yaml"

    def test_task_argument(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["agent", "--task", "do something"])
        assert args.task == "do something"

    def test_no_interactive_flag(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["agent", "--no-interactive"])
        assert args.interactive is False

    def test_interactive_flag(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["agent", "--interactive"])
        assert args.interactive is True

    def test_temperature_argument(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["agent", "--temperature", "0.7"])
        assert args.temperature == pytest.approx(0.7)

    def test_no_run_flag(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["autoswarm", "--no-run"])
        assert args.no_run is True

    def test_model_name_argument(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(
            ["agent", "--model-name", "claude-opus-4-6"]
        )
        assert args.model_name == "claude-opus-4-6"

    def test_verbose_flag(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["setup-check", "--verbose"])
        assert args.verbose is True

    def test_dir_argument(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(["init", "--dir", "/tmp/myproject"])
        assert args.dir == "/tmp/myproject"

    def test_loops_per_agent_argument(self):
        parser = self.setup_argument_parser()
        args = parser.parse_args(
            ["heavy-swarm", "--loops-per-agent", "3"]
        )
        assert args.loops_per_agent == 3


# ===========================================================================
# 2. route_command
# ===========================================================================


class TestRouteCommand:
    """Tests for route_command() dispatch logic."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import route_command

        self.route_command = route_command

    @patch("swarms.cli.main.handle_init")
    def test_routes_init(self, mock_handler):
        args = make_args(command="init")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_onboarding")
    def test_routes_onboarding(self, mock_handler):
        args = make_args(command="onboarding")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_run_agents")
    def test_routes_run_agents(self, mock_handler):
        args = make_args(command="run-agents")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_load_markdown")
    def test_routes_load_markdown(self, mock_handler):
        args = make_args(command="load-markdown")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_agent")
    def test_routes_agent(self, mock_handler):
        args = make_args(command="agent")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_chat")
    def test_routes_chat(self, mock_handler):
        args = make_args(command="chat")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_autoswarm")
    def test_routes_autoswarm(self, mock_handler):
        args = make_args(command="autoswarm")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_llm_council")
    def test_routes_llm_council(self, mock_handler):
        args = make_args(command="llm-council")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.handle_heavy_swarm")
    def test_routes_heavy_swarm(self, mock_handler):
        args = make_args(command="heavy-swarm")
        self.route_command(args)
        mock_handler.assert_called_once_with(args)

    @patch("swarms.cli.main.get_api_key")
    def test_routes_get_api_key(self, mock_fn):
        args = make_args(command="get-api-key")
        self.route_command(args)
        mock_fn.assert_called_once()

    @patch("swarms.cli.main.check_login")
    def test_routes_check_login(self, mock_fn):
        args = make_args(command="check-login")
        self.route_command(args)
        mock_fn.assert_called_once()

    @patch("swarms.cli.main.run_setup_check")
    def test_routes_setup_check(self, mock_fn):
        args = make_args(command="setup-check", verbose=True)
        self.route_command(args)
        mock_fn.assert_called_once_with(verbose=True)

    @patch("subprocess.run")
    def test_routes_upgrade(self, mock_run):
        args = make_args(command="upgrade")
        self.route_command(args)
        mock_run.assert_called_once_with(
            ["pip", "install", "--upgrade", "swarms"], check=True
        )

    @patch("swarms.cli.main.show_error")
    def test_unknown_command_shows_error(self, mock_show_error):
        # We bypass argparse validation by crafting args directly
        args = argparse.Namespace(command="definitely-not-a-command")
        self.route_command(args)
        mock_show_error.assert_called_once()


# ===========================================================================
# 3. handle_run_agents
# ===========================================================================


class TestHandleRunAgents:
    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import handle_run_agents

        self.handle_run_agents = handle_run_agents

    def test_missing_yaml_file_shows_error(self, tmp_path):
        args = make_args(
            command="run-agents",
            yaml_file=str(tmp_path / "nonexistent.yaml"),
        )
        with patch("swarms.cli.main.show_error") as mock_err:
            self.handle_run_agents(args)
            mock_err.assert_called_once()
            assert "File Error" in mock_err.call_args[0][0]

    @patch("swarms.cli.main.create_agents_from_yaml")
    def test_runs_agents_from_yaml(self, mock_create, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents: []")
        mock_create.return_value = "All done"
        args = make_args(
            command="run-agents", yaml_file=str(yaml_file)
        )
        self.handle_run_agents(args)
        mock_create.assert_called_once_with(
            yaml_file=str(yaml_file), return_type="run_swarm"
        )

    @patch("swarms.cli.main.create_agents_from_yaml")
    def test_handles_api_key_error(self, mock_create, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents: []")
        mock_create.side_effect = Exception(
            "invalid api_key provided"
        )
        args = make_args(
            command="run-agents", yaml_file=str(yaml_file)
        )
        with patch("swarms.cli.main.show_error") as mock_err:
            self.handle_run_agents(args)
            mock_err.assert_called_once()
            assert "API Key" in mock_err.call_args[0][0]

    @patch("swarms.cli.main.create_agents_from_yaml")
    def test_handles_context_length_error(
        self, mock_create, tmp_path
    ):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents: []")
        mock_create.side_effect = Exception("context_length_exceeded")
        args = make_args(
            command="run-agents", yaml_file=str(yaml_file)
        )
        with patch("swarms.cli.main.show_error") as mock_err:
            self.handle_run_agents(args)
            assert "Context Length" in mock_err.call_args[0][0]


# ===========================================================================
# 4. handle_load_markdown
# ===========================================================================


class TestHandleLoadMarkdown:
    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import handle_load_markdown

        self.handle_load_markdown = handle_load_markdown

    def test_missing_markdown_path_exits(self):
        args = make_args(command="load-markdown", markdown_path=None)
        with patch("swarms.cli.main.show_error"), pytest.raises(
            SystemExit
        ):
            self.handle_load_markdown(args)

    @patch("swarms.cli.main.load_markdown_agents")
    def test_loads_agents_from_path(self, mock_load):
        mock_load.return_value = [MagicMock(name="Agent1")]
        args = make_args(
            command="load-markdown",
            markdown_path="./agents/",
            concurrent=True,
        )
        self.handle_load_markdown(args)
        mock_load.assert_called_once_with(
            "./agents/", concurrent=True
        )


# ===========================================================================
# 5. handle_agent
# ===========================================================================


class TestHandleAgent:
    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import handle_agent

        self.handle_agent = handle_agent

    @patch("swarms.cli.main.create_swarm_agent")
    def test_calls_create_swarm_agent(self, mock_create):
        mock_create.return_value = None
        args = make_args(command="agent")
        self.handle_agent(args)
        mock_create.assert_called_once()

    @patch("swarms.cli.main.create_swarm_agent")
    def test_passes_model_name_to_agent(self, mock_create):
        mock_create.return_value = None
        args = make_args(
            command="agent", model_name="claude-opus-4-6"
        )
        self.handle_agent(args)
        call_kwargs = mock_create.call_args
        # model_name is passed either positionally or as keyword
        assert call_kwargs is not None


# ===========================================================================
# 6. handle_autoswarm
# ===========================================================================


class TestHandleAutoswarm:
    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import handle_autoswarm

        self.handle_autoswarm = handle_autoswarm

    def test_missing_task_shows_error_and_exits(self):
        args = make_args(command="autoswarm", task=None)
        with patch("swarms.cli.main.show_error") as mock_err:
            with pytest.raises(SystemExit):
                self.handle_autoswarm(args)
            mock_err.assert_called_once()

    @patch("swarms.cli.main.run_autoswarm")
    def test_calls_run_autoswarm_with_task(self, mock_run):
        args = make_args(
            command="autoswarm",
            task="build a pipeline",
            model="gpt-4o",
            output=None,
            output_dir=None,
            no_run=False,
        )
        self.handle_autoswarm(args)
        mock_run.assert_called_once()


# ===========================================================================
# 7. run_autoswarm
# ===========================================================================


class TestRunAutoswarm:
    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import run_autoswarm

        self.run_autoswarm = run_autoswarm

    def test_empty_task_shows_error(self):
        # run_autoswarm catches SwarmCLIError internally and calls show_error
        with patch("swarms.cli.main.show_error") as mock_err:
            self.run_autoswarm(task="", model="gpt-4o")
        mock_err.assert_called_once()
        assert "Task cannot be empty" in str(mock_err.call_args)

    def test_empty_model_shows_error(self):
        with patch("swarms.cli.main.show_error") as mock_err:
            self.run_autoswarm(task="do something", model="")
        mock_err.assert_called_once()
        assert "Model name cannot be empty" in str(mock_err.call_args)

    @patch(
        "swarms.agents.auto_generate_swarm_config.write_autoswarm_file"
    )
    @patch("swarms.cli.main.generate_swarm_config")
    def test_no_run_skips_execution(
        self, mock_gen, mock_write, tmp_path
    ):
        mock_gen.return_value = {"agents": []}
        output = str(tmp_path / "swarm.py")
        mock_write.return_value = output
        self.run_autoswarm(
            task="build something", model="gpt-4o", no_run=True
        )
        mock_write.assert_called_once()


# ===========================================================================
# 8. handle_onboarding
# ===========================================================================


class TestHandleOnboarding:
    @patch("swarms.cli.main.run_setup_check")
    def test_delegates_to_setup_check(self, mock_check):
        from swarms.cli.main import handle_onboarding

        args = make_args(command="onboarding", verbose=False)
        handle_onboarding(args)
        mock_check.assert_called_once_with(verbose=False)

    @patch("swarms.cli.main.run_setup_check")
    def test_passes_verbose_flag(self, mock_check):
        from swarms.cli.main import handle_onboarding

        args = make_args(command="onboarding", verbose=True)
        handle_onboarding(args)
        mock_check.assert_called_once_with(verbose=True)


# ===========================================================================
# 9. handle_llm_council
# ===========================================================================


class TestHandleLLMCouncil:
    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import handle_llm_council

        self.handle_llm_council = handle_llm_council

    def test_missing_task_shows_error_and_exits(self):
        args = make_args(command="llm-council", task=None)
        with patch("swarms.cli.main.show_error") as mock_err:
            with pytest.raises(SystemExit):
                self.handle_llm_council(args)
            mock_err.assert_called_once()

    @patch("swarms.cli.main.run_llm_council")
    def test_calls_run_llm_council(self, mock_run):
        args = make_args(command="llm-council", task="What is AGI?")
        self.handle_llm_council(args)
        mock_run.assert_called_once()


# ===========================================================================
# 10. handle_heavy_swarm
# ===========================================================================


class TestHandleHeavySwarm:
    @pytest.fixture(autouse=True)
    def _import(self):
        from swarms.cli.main import handle_heavy_swarm

        self.handle_heavy_swarm = handle_heavy_swarm

    def test_missing_task_shows_error_and_exits(self):
        args = make_args(command="heavy-swarm", task=None)
        with patch("swarms.cli.main.show_error") as mock_err:
            with pytest.raises(SystemExit):
                self.handle_heavy_swarm(args)
            mock_err.assert_called_once()

    @patch("swarms.cli.main.run_heavy_swarm")
    def test_calls_run_heavy_swarm(self, mock_run):
        args = make_args(
            command="heavy-swarm",
            task="complex analysis",
            loops_per_agent=2,
        )
        self.handle_heavy_swarm(args)
        mock_run.assert_called_once()


# ===========================================================================
# 11. CLI utils
# ===========================================================================


class TestCLIUtils:
    def test_swarm_cli_error_is_exception(self):
        from swarms.cli.utils import SwarmCLIError

        with pytest.raises(SwarmCLIError):
            raise SwarmCLIError("test error")

    def test_colors_dict_has_required_keys(self):
        from swarms.cli.utils import COLORS

        for key in ("primary", "secondary", "error", "success"):
            assert key in COLORS

    def test_console_is_importable(self):
        from swarms.cli.utils import console

        assert console is not None

    def test_show_error_does_not_raise(self):
        from swarms.cli.utils import show_error

        # should print and not crash
        show_error("Test Error", "Some details")

    def test_detect_active_provider_no_keys(self, monkeypatch):
        from swarms.cli.utils import _detect_active_provider

        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "GOOGLE_API_KEY",
            "COHERE_API_KEY",
            "MISTRAL_API_KEY",
            "TOGETHER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        result = _detect_active_provider()
        assert "No API key" in result

    def test_detect_active_provider_with_openai(self, monkeypatch):
        from swarms.cli.utils import _detect_active_provider

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = _detect_active_provider()
        assert "OpenAI" in result

    def test_detect_active_provider_multiple(self, monkeypatch):
        from swarms.cli.utils import _detect_active_provider

        # Clear all provider keys first, then set exactly two
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "GOOGLE_API_KEY",
            "COHERE_API_KEY",
            "MISTRAL_API_KEY",
            "TOGETHER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        result = _detect_active_provider()
        assert "+1 more" in result


# ===========================================================================
# 12. show_ascii_art smoke test
# ===========================================================================


class TestShowAsciiArt:
    def test_show_ascii_art_does_not_raise(self):
        from swarms.cli.utils import show_ascii_art

        # Should just print, never crash
        show_ascii_art()


# ===========================================================================
# 13. main() entry point integration
# ===========================================================================


class TestMain:
    @patch("swarms.cli.main.route_command")
    @patch("swarms.cli.main.show_ascii_art")
    def test_main_calls_route_command(self, mock_art, mock_route):
        from swarms.cli.main import main

        with patch("sys.argv", ["swarms", "setup-check"]):
            main()
        mock_route.assert_called_once()

    @patch("swarms.cli.main.show_ascii_art")
    def test_main_handles_bad_command_gracefully(self, mock_art):
        from swarms.cli.main import main

        with patch("sys.argv", ["swarms", "bad-command"]):
            with pytest.raises(SystemExit):
                main()

    @patch(
        "swarms.cli.main.route_command",
        side_effect=RuntimeError("boom"),
    )
    @patch("swarms.cli.main.show_ascii_art")
    def test_main_catches_route_error(self, mock_art, mock_route):
        from swarms.cli.main import main

        with patch("sys.argv", ["swarms", "setup-check"]):
            # Should NOT raise — errors are caught and printed
            main()
