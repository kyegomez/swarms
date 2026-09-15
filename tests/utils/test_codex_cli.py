"""Offline Codex transport, Agent, YAML, and workflow integration tests."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from swarms import Agent, SequentialWorkflow
from swarms.agents.context_compressor import ContextCompressor
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.utils import codex_cli
from swarms.utils.codex_cli import CodexCLI
from swarms.utils.litellm_wrapper import LiteLLM


@pytest.fixture
def transport(monkeypatch):
    state = SimpleNamespace(
        answer="سلام",
        returncode=0,
        error=None,
        missing=False,
        calls=[],
    )
    monkeypatch.setattr(
        codex_cli.shutil, "which", lambda name: "/test/codex"
    )

    def popen(command, **options):
        process = Mock(pid=12345, returncode=state.returncode)

        def communicate(prompt=None, timeout=None):
            if prompt is None:
                return "", ""
            state.calls.append((command, options, prompt, timeout))
            if state.error:
                raise state.error
            if not state.missing:
                output = Path(
                    command[
                        command.index("--output-last-message") + 1
                    ]
                )
                output.write_text(state.answer, encoding="utf-8")
            return (
                "progress is not the final answer",
                "authentication required",
            )

        process.communicate.side_effect = communicate
        context = Mock()
        context.__enter__ = Mock(return_value=process)
        context.__exit__ = Mock(return_value=False)
        state.process = process
        return context

    monkeypatch.setattr(codex_cli.subprocess, "Popen", popen)
    return state


def agent(**kwargs):
    return Agent(
        llm_backend="codex",
        model_name=None,
        print_on=False,
        output_type="final",
        retry_attempts=1,
        **kwargs,
    )


def payload(call):
    return json.loads(call[2].split("\n", 1)[1])


def test_text_transport_and_isolated_defaults(transport):
    llm = CodexCLI(system_prompt="Answer in Persian.", timeout=12)
    assert llm.run("سلام؛ $HOME `echo never` 🐍") == "سلام"
    command, options, prompt, timeout = transport.calls[0]
    assert "--model" not in command
    assert command[-1] == "-"
    assert "--ignore-user-config" in command
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "features.shell_tool=false" in command
    assert options["encoding"] == "utf-8"
    assert not options.get("shell")
    assert not Path(options["cwd"]).exists()
    assert timeout == 12
    assert "سلام" in prompt
    assert "$HOME" not in " ".join(command)
    assert (
        payload(transport.calls[0])["system_prompt"]
        == "Answer in Persian."
    )


def test_structured_history_and_model_selection(transport):
    messages = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hello Ada."},
    ]
    CodexCLI(model_name="codex-model").run(
        task="What is my name?", messages=messages
    )
    command = transport.calls[0][0]
    assert command[command.index("--model") + 1] == "codex-model"
    sent = payload(transport.calls[0])["messages"]
    assert sent[:2] == messages
    assert sent[-1]["content"] == "What is my name?"
    assert len(messages) == 2


def test_agent_forwards_history_and_updated_instructions(transport):
    assistant = agent(system_prompt="Initial instructions")
    assistant.run("First question")
    assistant.update_system_prompt("New instructions")
    assistant.run("Second question")
    sent = payload(transport.calls[-1])
    assert sent["system_prompt"] == "New instructions"
    assert [m["content"] for m in sent["messages"]] == [
        "First question",
        "سلام",
        "Second question",
    ]


def test_yaml_and_two_agent_workflow(transport):
    config = {
        "agents": [
            {
                "agent_name": name,
                "system_prompt": prompt,
                "llm_backend": "codex",
                "model_name": None,
                "max_loops": 1,
                "autosave": False,
                "print_on": False,
                "output_type": "final",
                "codex_config": {"timeout": 15},
            }
            for name, prompt in [
                ("Writer", "Write a draft"),
                ("Reviewer", "Review it"),
            ]
        ]
    }
    agents = create_agents_from_yaml(
        yaml_string=yaml.safe_dump(config), return_type="agents"
    )
    result = SequentialWorkflow(
        agents=agents, output_type="final", autosave=False
    ).run("Explain CSV files")
    assert result == "سلام"
    assert len(transport.calls) == 2
    assert transport.calls[1][3] == 15
    assert "سلام" in transport.calls[1][2]


def test_compression_stays_on_codex(transport, monkeypatch):
    completion = Mock(
        side_effect=AssertionError("Must not call an API provider")
    )
    monkeypatch.setattr(
        "swarms.agents.context_compressor.completion", completion
    )
    assistant = agent(system_prompt="Original instructions")
    assert (
        ContextCompressor()._summarize(assistant, "Remember Ada")
        == "سلام"
    )
    assert "Remember Ada" in transport.calls[0][2]
    assert (
        "compression expert"
        in payload(transport.calls[0])["system_prompt"]
    )
    assert assistant.llm.system_prompt == "Original instructions"
    completion.assert_not_called()


def test_existing_backend_and_custom_llm_unchanged():
    original = Agent(print_on=False)
    restored = Agent(
        llm_backend=original.llm_backend,
        codex_config=original.codex_config,
        print_on=False,
    )
    assert isinstance(restored.llm, LiteLLM)
    fake = Mock()
    assert Agent(llm=fake, print_on=False).llm is fake


@pytest.mark.parametrize(
    "timeout", [0, -1, float("inf"), float("nan"), True, "10"]
)
def test_invalid_timeout(timeout, transport):
    with pytest.raises(ValueError, match="timeout"):
        agent(codex_config={"timeout": timeout})
    assert not transport.calls


@pytest.mark.parametrize(
    "config",
    [
        {"llm_backend": "unknown"},
        {"codex_config": {}},
        {"llm_backend": "codex", "codex_config": []},
        {"llm_backend": "codex", "llm": Mock()},
    ],
)
def test_invalid_backend_configuration(config, transport):
    with pytest.raises(ValueError):
        Agent(**config)


def test_unknown_codex_option(transport):
    with pytest.raises(TypeError, match="unexpected keyword"):
        agent(codex_config={"timeuot": 10})


@pytest.mark.parametrize(
    "option,value",
    [
        ("tools", [lambda: None]),
        ("tools_list_dictionary", [{"type": "function"}]),
        ("mcp_url", "https://example.invalid/mcp"),
        ("stream", True),
        ("streaming_on", True),
        ("multi_modal", True),
        ("max_loops", "auto"),
        ("handoffs", ["other-agent"]),
        ("think_tool", True),
        ("llm_args", {"tools": []}),
        ("llm_base_url", "https://example.invalid"),
    ],
)
def test_unsupported_agent_options(option, value, transport):
    with pytest.raises(ValueError, match="unsupported options"):
        agent(**{option: value})
    assert not transport.calls


def test_runtime_streaming_toggle_rejected(transport):
    assistant = agent()
    assistant.stream = True
    with pytest.raises(ValueError, match="unsupported options"):
        assistant.call_llm("hello")
    assert not transport.calls


def test_codex_capabilities_do_not_use_litellm_catalog(
    transport, monkeypatch
):
    capability_lookup = Mock(
        side_effect=AssertionError("Not a LiteLLM model")
    )
    monkeypatch.setattr(
        "swarms.agents.llm_manager.supports_function_calling",
        capability_lookup,
    )
    assistant = agent()
    assert assistant.run("Hello") == "سلام"
    capability_lookup.assert_not_called()
    with pytest.raises(ValueError, match="images"):
        assistant.check_model_supports_utilities(img="picture.png")
    with pytest.raises(ValueError, match="streaming callbacks"):
        assistant.call_llm(
            "Hello", streaming_callback=lambda text: None
        )


def test_codex_model_rotation_stays_on_backend(transport):
    assistant = agent(fallback_models=["codex-a", "codex-b"])
    assert assistant.llm.model_name == "codex-a"
    assert assistant.switch_to_next_model()
    assert isinstance(assistant.llm, CodexCLI)
    assert assistant.llm.model_name == "codex-b"


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"task": " "},
        {"task": "hi", "img": "picture.png"},
        {"task": "hi", "stream": True},
        {"messages": [{"role": "tool", "content": "result"}]},
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url"}]}
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{}],
                }
            ]
        },
    ],
)
def test_unsupported_input(inputs, transport):
    with pytest.raises(ValueError):
        CodexCLI().run(**inputs)
    assert not transport.calls


def test_missing_cli(monkeypatch):
    monkeypatch.setattr(codex_cli.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="codex login"):
        agent()


@pytest.mark.parametrize(
    "missing,answer", [(True, "unused"), (False, "  ")]
)
def test_empty_response(missing, answer, transport):
    transport.missing, transport.answer = missing, answer
    with pytest.raises(RuntimeError, match="without a final"):
        CodexCLI().run("Hello")
    assert not Path(transport.calls[0][1]["cwd"]).exists()


def test_cli_auth_failure(transport):
    transport.returncode = 1
    with pytest.raises(RuntimeError, match="codex login status"):
        CodexCLI().run("Hello")
    assert not Path(transport.calls[0][1]["cwd"]).exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_timeout_and_cancellation_cleanup(
    transport, monkeypatch, interrupted
):
    transport.error = (
        KeyboardInterrupt()
        if interrupted
        else subprocess.TimeoutExpired("codex", 1)
    )
    child = Mock()
    parent = Mock()
    parent.children.return_value = [child]
    monkeypatch.setattr(
        codex_cli.psutil, "Process", lambda pid: parent
    )
    expected = KeyboardInterrupt if interrupted else TimeoutError
    with pytest.raises(expected):
        CodexCLI(timeout=1).run("Hello")
    child.kill.assert_called_once()
    transport.process.kill.assert_called_once()
    assert not Path(transport.calls[0][1]["cwd"]).exists()
