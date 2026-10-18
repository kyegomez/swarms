"""Check Lumo routing with real Agent/LLMManager methods offline."""

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


def test_example_routes_generation_to_local_lumo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Mock(
        return_value=SimpleNamespace(
            run=Mock(return_value="wrong provider route")
        )
    )
    manager = load_methods(
        "swarms/agents/llm_manager.py",
        "LLMManager",
        {
            "__init__",
            "build",
            "_call_llm",
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
    outputs = []

    def one_model_call(self: object, task: str) -> str:
        result = self.llm_manager._call_llm(task)
        outputs.append(result)
        return result

    agent.run = one_model_call
    model = SimpleNamespace(
        device="cpu", generate=Mock(return_value=[[101, 102]])
    )
    inputs = Mock()
    inputs.to.return_value = {"input_ids": [[101]]}
    tokenizer = Mock(return_value=inputs)
    tokenizer.decode.return_value = "local Lumo answer"
    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=agent)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float16="offline-float16"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            BitsAndBytesConfig=Mock(),
            LlamaForCausalLM=SimpleNamespace(
                from_pretrained=Mock(return_value=model)
            ),
            AutoTokenizer=SimpleNamespace(
                from_pretrained=Mock(return_value=tokenizer)
            ),
        ),
    )
    blocked = Mock(side_effect=AssertionError("No network or input"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("builtins.input", blocked)
    runpy.run_path(str(ROOT / "examples/models/lumo/lumo_example.py"))

    provider.assert_not_called()
    assert outputs == ["local Lumo answer"]
    tokenizer.assert_called_once_with(
        "How do i create a smart contract in solana?",
        return_tensors="pt",
    )
    inputs.to.assert_called_once_with("cpu")
    model.generate.assert_called_once_with(
        input_ids=[[101]], max_new_tokens=100
    )
