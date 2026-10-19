import importlib.util
import runpy
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/multi_agent/council/judges"


def load_module(monkeypatch, name: str) -> ModuleType:
    """Load checked-in code without importing provider packages."""
    path = ROOT / (name.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def council_runtime(monkeypatch, tmp_path: Path) -> list:
    """Use the real council, conversation, context and formatter."""
    monkeypatch.chdir(tmp_path)
    no_network = Mock(side_effect=AssertionError("Network forbidden"))
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    for name in ("swarms", "swarms.structs", "swarms.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    def stub(name: str, **attributes) -> None:
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    calls = []

    class OfflineAgent:
        """Return deterministic generation and evaluation results."""

        def __init__(self, agent_name: str, **kwargs) -> None:
            self.agent_name = agent_name

        def run(self, task: str, messages=None) -> str:
            calls.append((self.agent_name, task, messages))
            if self.agent_name == "Financial-Analysis-Agent":
                return "Generated financial answer."
            if self.agent_name == "aggregator_agent":
                return "Aggregated report."
            return f"{self.agent_name}: evaluated."

    stub("loguru", logger=Mock())
    stub("swarms.structs.agent", Agent=OfflineAgent)
    stub(
        "swarms.structs.ma_utils",
        set_random_models_for_agents=Mock(
            side_effect=AssertionError("Unexpected random model")
        ),
    )
    stub(
        "swarms.utils.litellm_tokenizer",
        count_tokens=lambda text, **kwargs: len(str(text).split()),
    )
    stub(
        "swarms.telemetry.otel",
        ContextThreadPoolExecutor=ThreadPoolExecutor,
        capture_init=lambda instance: None,
        trace_run=lambda *args, **kwargs: lambda function: function,
    )
    for name in (
        "swarms.utils.generate_id",
        "swarms.utils.output_types",
        "swarms.structs.conversation",
        "swarms.structs.context_utils",
        "swarms.utils.history_output_formatter",
        "swarms.structs.council_as_judge",
    ):
        load_module(monkeypatch, name)
    yield calls
    no_network.assert_not_called()


@pytest.mark.parametrize(
    ("filename", "evaluation_index"),
    [
        ("council_judge_basic.py", None),
        ("council_of_judges_eval.py", None),
        ("council_judge_example.py", None),
        *[("council_judge_custom_example.py", i) for i in range(3)],
        *[("council_judge_complex_example.py", i) for i in range(3)],
    ],
)
def test_council_examples_evaluate_responses(
    council_runtime: list, filename: str, evaluation_index
) -> None:
    """Keep all six judges and the aggregator on the example dataflow."""
    generated_answer = filename in (
        "council_judge_basic.py",
        "council_of_judges_eval.py",
    )
    namespace = runpy.run_path(
        str(EXAMPLES / filename),
        run_name="__main__" if generated_answer else "example",
    )
    if generated_answer:
        result = namespace["results"]
    elif evaluation_index is None:
        result = namespace["main"]()
    else:
        evaluations = [
            value
            for name, value in namespace.items()
            if name.startswith("evaluate_") and callable(value)
        ]
        result = evaluations[evaluation_index]()

    judges = [
        call for call in council_runtime if call[0].endswith("_judge")
    ]
    assert len(judges) == 6
    assert len({call[0] for call in judges}) == 6
    assert all("Task:" in call[1] for call in judges)
    assert all("Response:" in call[1] for call in judges)
    if generated_answer:
        assert council_runtime[0][0] == "Financial-Analysis-Agent"
        assert all(
            "Generated financial answer." in c[1] for c in judges
        )
    name, task, messages = council_runtime[-1]
    assert name == "aggregator_agent"
    assert len(messages) == 6
    aggregation_context = str(messages) + task
    assert all(call[0] in aggregation_context for call in judges)
    if (
        filename == "council_judge_custom_example.py"
        and evaluation_index == 1
    ):
        assert isinstance(result, list)
    if isinstance(result, list):
        assert len(result) == 8
        assert result[-1]["content"] == "Aggregated report."
    else:
        assert "Aggregated report." in result
