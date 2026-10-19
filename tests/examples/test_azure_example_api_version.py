"""Verify Azure configuration with real Agent/LLMManager methods."""

import ast
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_methods(
    path: str, name: str, methods: set[str], namespace: dict
) -> type:
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - Execute only the checked-in class methods.
        compile(ast.fix_missing_locations(module), path, "exec"),
        namespace,
    )
    return namespace[name]


def test_azure_example_retains_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Mock()
    manager = load_methods(
        "swarms/agents/llm_manager.py",
        "LLMManager",
        {
            "__init__",
            "build",
            "get_current_model",
            "get_available_models",
        },
        {
            "LiteLLM": provider,
            "exists": lambda value: value is not None,
            "DEFAULT_MODEL_NAME": "offline-provider",
            "AgentLLMInitializationError": RuntimeError,
            "logger": Mock(),
        },
    )
    helpers = {
        name: Mock()
        for name in (
            "SkillsManager",
            "ContextCompressor",
            "MCPManager",
            "AgentMarketplaceHandler",
            "ToolManager",
            "AutonomousAgentLoop",
            "capture_init",
        )
    }
    helpers.update(
        LLMManager=manager,
        generate_id=lambda prefix: "offline-agent",
        generate_api_key=lambda **kwargs: "offline-state",
        get_workspace_dir=lambda: "unused-offline-workspace",
        build_agent_system_prompt=lambda: "Offline system",
        get_autonomous_agent_prompt=lambda **kwargs: "Offline plan",
        empty_usage=dict,
        MAX_PLANNING_ATTEMPTS=1,
        MAX_SUBTASK_ITERATIONS=1,
        MAX_SUBTASK_LOOPS=1,
    )
    agent = load_methods(
        "swarms/structs/agent.py",
        "Agent",
        {"__init__", "llm_handling"},
        helpers,
    )
    for name in (
        "setup_config",
        "short_memory_init",
        "reliability_check",
        "_add_usage",
    ):
        setattr(agent, name, Mock())
    agent._default_context_length = Mock(return_value=8192)
    agent._default_max_tokens = Mock(return_value=1000)
    agent.mcp_enabled = False
    agent.run = Mock(return_value="offline answer")
    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=agent)
    )
    blocked = Mock(side_effect=AssertionError("No network or input"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("builtins.input", blocked)
    namespace = runpy.run_path(
        str(ROOT / "examples/models/azure/azure_agent_api_verison.py")
    )

    assert provider.call_args.kwargs["model_name"] == "azure/gpt-4.1"
    assert provider.call_args.kwargs.get("api_version") == "...."
    provider.reset_mock()
    namespace["agent"].llm_manager.build()
    assert provider.call_args.kwargs.get("api_version") == "...."
