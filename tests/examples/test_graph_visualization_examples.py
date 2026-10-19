"""Exercise visualization examples with real graphs and offline I/O"""

import io
import runpy
import socket
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

graphviz = pytest.importorskip("graphviz")
pytest.importorskip("networkx")
ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/multi_agent/graphworkflow_examples"


class OfflineAgent:
    """Supply node names without constructing a model client."""

    def __init__(self, agent_name: str, **kwargs) -> None:
        self.agent_name = agent_name

    def run(self, *args, **kwargs) -> None:
        raise AssertionError("Visualization must not run agents")


@pytest.fixture
def graph_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load actual graph/conversation modules with offline helpers."""
    names = [
        "swarms",
        "swarms.structs",
        "swarms.utils",
        "swarms.structs.agent",
        "swarms.structs.conversation",
        "swarms.structs.graph_workflow",
        "swarms.utils.get_cpu_cores",
        "swarms.utils.loguru_logger",
        "swarms.telemetry",
        "swarms.telemetry.otel",
        "swarms.utils.generate_id",
        "swarms.utils.litellm_wrapper",
        "swarms.utils.litellm_tokenizer",
        "loguru",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["swarms"].Agent = OfflineAgent
    modules["swarms.structs.agent"].Agent = OfflineAgent
    modules["swarms.utils.get_cpu_cores"].get_cpu_cores = lambda: 2
    modules["swarms.utils.loguru_logger"].initialize_logger = (
        lambda **kwargs: Mock()
    )
    telemetry = modules["swarms.telemetry.otel"]
    telemetry.ContextThreadPoolExecutor = ThreadPoolExecutor
    telemetry.capture_init = Mock()
    telemetry.trace_run = lambda *args, **kwargs: lambda fn: fn
    modules["swarms.utils.generate_id"].generate_id = (
        lambda *args: uuid.uuid4().hex
    )
    modules["swarms.utils.litellm_wrapper"].empty_usage = dict
    modules["swarms.utils.litellm_tokenizer"].count_tokens = Mock(
        return_value=0
    )
    modules["loguru"].logger = Mock()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        socket,
        "socket",
        Mock(side_effect=AssertionError("Network is forbidden")),
    )
    for name in ("conversation", "graph_workflow"):
        path = ROOT / f"swarms/structs/{name}.py"
        module = modules[f"swarms.structs.{name}"]
        exec(  # noqa: S102 - execute the actual local module
            compile(path.read_text(), str(path), "exec"),
            module.__dict__,
        )
    return modules["swarms.structs.graph_workflow"]


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list:
    """Keep actual DOT generation; replace external rendering only."""
    calls = []

    def render(
        dot: graphviz.Digraph,
        filename: str,
        view: bool = False,
        cleanup: bool = False,
    ) -> str:
        calls.append((dot, filename, view, cleanup))
        return f"{filename}.{dot.format}"

    monkeypatch.setattr(graphviz.Digraph, "render", render)
    return calls


def run_example(
    filename: str, run_name: str = "__main__"
) -> tuple[dict, str]:
    """Execute the complete example and capture its diagnostics."""
    output = io.StringIO()
    with redirect_stdout(output):
        namespace = runpy.run_path(
            str(EXAMPLES / filename), run_name=run_name
        )
    return namespace, output.getvalue()


def test_comprehensive_visualization(
    graph_module: ModuleType, renders: list
) -> None:
    _, output = run_example("test_graphviz_visualization.py")
    assert "failed:" not in output
    assert len(renders) == 12
    assert [call[0].engine for call in renders[:5]] == [
        "dot",
        "neato",
        "fdp",
        "sfdp",
        "circo",
    ]
    assert [call[0].format for call in renders[5:9]] == [
        "png",
        "svg",
        "pdf",
        "dot",
    ]
    assert all(
        not view and cleanup for _, _, view, cleanup in renders
    )
    assert len({filename for _, filename, _, _ in renders}) == 12
    assert "WebScraper -> TextProcessor" in renders[0][0].source
    assert "Fan-out:" in renders[0][0].source
    assert "DataCollector -> DataAnalyzer" in renders[5][0].source
    assert "DataIngestionAgent1" in renders[-1][0].source
    assert "Nodes: 12, Edges: 19" in output


def test_parallel_summary_toggle(
    graph_module: ModuleType, renders: list
) -> None:
    namespace, _ = run_example(
        "test_graphviz_visualization.py", "example"
    )
    output = io.StringIO()
    with redirect_stdout(output):
        demo = next(
            fn
            for name, fn in namespace.items()
            if name.startswith("test_parallel_pattern_")
        )
        demo()
    assert len(renders) == 2
    assert renders[0][0].source == renders[1][0].source
    assert (
        output.getvalue().count("GRAPHVIZ WORKFLOW VISUALIZATION")
        == 1
    )


def test_text_visualization_without_graphviz(
    graph_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _ = run_example(
        "test_graphviz_visualization.py", "example"
    )
    monkeypatch.setattr(graph_module, "GRAPHVIZ_AVAILABLE", False)
    with patch.object(
        graphviz.Digraph, "render", side_effect=AssertionError
    ):
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["test_fallback_visualization"]()
    assert "failed:" not in output.getvalue()
    assert "WebScraper → TextProcessor" in output.getvalue()
    assert "Fan-in patterns:" in output.getvalue()


def test_parallel_workflow_entrypoint(
    graph_module: ModuleType, renders: list
) -> None:
    namespace, output = run_example(
        "test_parallel_processing_example.py"
    )
    assert len(renders) == 1
    dot, filename, view, cleanup = renders[0]
    assert dot.format == "png" and view and cleanup
    assert filename.startswith(
        "Advanced-Parallel-Financial-Analysis_"
    )
    workflow = namespace["workflow"]
    assert len(workflow.nodes) == 10
    for edge in workflow.edges:
        assert f"{edge.source} -> {edge.target}" in dot.source
    assert "GRAPHVIZ WORKFLOW VISUALIZATION" in output
