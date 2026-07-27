import json
import os

import pytest
from dotenv import load_dotenv
from opentelemetry.sdk.trace.export import (  # noqa: E402
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from swarms.telemetry.otel import (  # noqa: E402
    MAX_PAYLOAD_CHARS,
    SwarmTelemetry,
    _SpanHandle,
    _truncate,
    capture_error,
    capture_init,
    capture_run,
    init_config,
    log_agent_data,
    swarm_telemetry,
    telemetry_on,
    trace_run,
)

os.environ["SWARMS_OTEL_TIMEOUT"] = "2"

load_dotenv()

# Real-run tests need a live LLM key; skip them (only them) when none is set.
_LLM_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)
_HAS_LLM_KEY = any(os.getenv(k) for k in _LLM_KEYS)
_LLM_MODEL = os.getenv("SWARMS_TEST_MODEL", "gpt-4o-mini")

requires_llm = pytest.mark.skipif(
    not _HAS_LLM_KEY,
    reason="no LLM API key set (OPENAI_API_KEY etc.) — real-run tests skipped",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def _exporter():
    """Build the telemetry singleton and attach an in-memory span exporter.

    The receiver URL is hard-coded in the codebase, so for the test run we point
    the module constant at a dead local address before the provider is built —
    no real telemetry ever leaves the machine.
    """
    import swarms.telemetry.otel as _otel

    _otel.TELEMETRY_BASE_URL = "http://127.0.0.1:9/telemetry-test"
    swarm_telemetry.cache_clear()
    telem = swarm_telemetry()
    assert (
        telem.ready
    ), "telemetry must be ready with SWARMS_TELEMETRY_ON=true"
    mem = InMemorySpanExporter()
    telem._provider.add_span_processor(SimpleSpanProcessor(mem))
    return mem


@pytest.fixture
def spans(_exporter):
    """Clear captured spans before each test; yield the exporter for reading."""
    _exporter.clear()
    return _exporter


def _by_name(exporter, name):
    return next(
        (s for s in exporter.get_finished_spans() if s.name == name),
        None,
    )


def _attrs(span):
    return dict(span.attributes)


def _agent(name="TelemetryTestAgent"):
    from swarms import Agent

    return Agent(
        agent_name=name,
        model_name=_LLM_MODEL,
        max_loops=1,
        max_tokens=200,
        persistent_memory=False,
        print_on=False,
        verbose=False,
        system_prompt="You are a terse assistant. Answer in one short sentence.",
    )


# ===========================================================================
# Pure helpers
# ===========================================================================
class TestHelpers:
    def test_telemetry_on_truthy(self, monkeypatch):
        # anything that is not a recognized off value means on
        for val in ("true", "True", "1", "yes", "on", "anything"):
            monkeypatch.setenv("SWARMS_TELEMETRY_ON", val)
            assert telemetry_on() is True

    def test_telemetry_on_falsy(self, monkeypatch):
        for val in (
            "false",
            "False",
            "FALSE",
            "0",
            "no",
            "off",
            "disable",
            "disabled",
            " false ",
        ):
            monkeypatch.setenv("SWARMS_TELEMETRY_ON", val)
            assert telemetry_on() is False

    def test_telemetry_on_unset_defaults_to_on(self, monkeypatch):
        """Telemetry is opt-out: unset means enabled."""
        monkeypatch.delenv("SWARMS_TELEMETRY_ON", raising=False)
        assert telemetry_on() is True

    def test_telemetry_on_empty_value_is_off(self, monkeypatch):
        """`SWARMS_TELEMETRY_ON=` in a .env disables rather than enables."""
        for val in ("", "   "):
            monkeypatch.setenv("SWARMS_TELEMETRY_ON", val)
            assert telemetry_on() is False

    def test_truncate_short_passthrough(self):
        assert _truncate("hello") == "hello"

    def test_truncate_non_string(self):
        assert _truncate(12345) == "12345"

    def test_truncate_long_marks(self):
        out = _truncate("x" * (MAX_PAYLOAD_CHARS + 50))
        assert out.endswith("…[truncated]")
        assert len(out) <= MAX_PAYLOAD_CHARS + len("…[truncated]")

    def test_init_config_only_constructor_params(self):
        class Thing:
            def __init__(self, alpha=1, beta="b"):
                self.alpha = alpha
                self.beta = beta
                self._internal = "hidden"  # not an __init__ param

        cfg = json.loads(init_config(Thing()))
        assert cfg == {"alpha": 1, "beta": "b"}
        assert "_internal" not in cfg

    def test_init_config_non_json_falls_back_to_str(self):
        obj = object()

        class Holder:
            def __init__(self, thing=None):
                self.thing = thing

        cfg = json.loads(init_config(Holder(obj)))
        assert isinstance(
            cfg["thing"], str
        )  # str() fallback, not a crash


# ===========================================================================
# _SpanHandle
# ===========================================================================
class TestSpanHandle:
    def test_noop_handle_never_raises(self):
        h = _SpanHandle(None)
        h.set("k", "v")
        h.record_output("out")
        h.record_error(ValueError("x"))  # must not raise

    def test_output_then_error_first_wins(self, spans):
        with capture_run("Idem.run", None) as h:
            h.record_output("done")
            h.record_error(
                ValueError("late")
            )  # ignored (already done)
        span = _by_name(spans, "Idem.run")
        assert _attrs(span)["swarms.status"] == "completed"


# ===========================================================================
# capture_run — input / output / auto error capture
# ===========================================================================
class TestCaptureRun:
    def test_captures_inputs(self, spans):
        with capture_run(
            "Op.run", None, task="analyze", img=None
        ) as h:
            h.record_output("ok")
        a = _attrs(_by_name(spans, "Op.run"))
        assert a["swarms.input.task"] == "analyze"
        assert "swarms.input.img" not in a  # None inputs skipped

    def test_captures_output_and_status(self, spans):
        with capture_run("Op.run", None) as h:
            h.record_output("the answer")
        a = _attrs(_by_name(spans, "Op.run"))
        assert a["swarms.output"] == "the answer"
        assert a["swarms.status"] == "completed"

    def test_auto_captures_propagating_error(self, spans):
        with pytest.raises(ValueError):
            with capture_run("Op.run", None, task="boom"):
                raise ValueError("kaboom")
        span = _by_name(spans, "Op.run")
        a = _attrs(span)
        assert a["swarms.status"] == "error"
        assert a["swarms.error.type"] == "ValueError"
        assert a["swarms.error.message"] == "kaboom"
        assert span.status.status_code.name == "ERROR"
        assert any(e.name == "exception" for e in span.events)

    def test_error_recorded_once(self, spans):
        with pytest.raises(RuntimeError):
            with capture_run("Op.run", None):
                raise RuntimeError("once")
        span = _by_name(spans, "Op.run")
        assert (
            len([e for e in span.events if e.name == "exception"])
            == 1
        )


# ===========================================================================
# capture_error — swallowed errors
# ===========================================================================
class TestCaptureError:
    def test_emits_error_span_with_context(self, spans):
        try:
            raise KeyError("agent-3 failed")
        except Exception as e:
            capture_error(
                e,
                None,
                name="ConcurrentWorkflow.agent_error",
                agent="Worker-3",
            )
        span = _by_name(spans, "ConcurrentWorkflow.agent_error")
        a = _attrs(span)
        assert a["swarms.status"] == "error"
        assert a["swarms.error.type"] == "KeyError"
        assert a["swarms.agent"] == "Worker-3"
        assert any(e.name == "exception" for e in span.events)

    def test_default_span_name(self, spans):
        class Widget:
            pass

        capture_error(ValueError("x"), Widget())
        assert _by_name(spans, "Widget.error") is not None


# ===========================================================================
# trace_run decorator
# ===========================================================================
class TestTraceRun:
    def test_records_output(self, spans):
        class Comp:
            agent_name = "Comp-A"
            id = "c1"

            @trace_run("Comp.run", input_params=("task",))
            def run(self, task=None):
                return f"result:{task}"

        assert Comp().run(task="hi") == "result:hi"
        a = _attrs(_by_name(spans, "Comp.run"))
        assert a["swarms.input.task"] == "hi"
        assert a["swarms.output"] == "result:hi"
        assert a["swarms.status"] == "completed"

    def test_records_and_reraises_error(self, spans):
        class Comp:
            @trace_run("Comp.boom")
            def run(self, task=None):
                raise ValueError("bad")

        with pytest.raises(ValueError):
            Comp().run(task="x")
        a = _attrs(_by_name(spans, "Comp.boom"))
        assert a["swarms.status"] == "error"
        assert a["swarms.error.type"] == "ValueError"

    def test_preserves_wrapped(self):
        class Comp:
            @trace_run("Comp.run")
            def run(self):
                return 1

        assert hasattr(Comp.run, "__wrapped__")


# ===========================================================================
# Identity schema — accessible swarms.* namespace
# ===========================================================================
class TestSchema:
    def test_agent_identity_no_swarm_type(self, spans):
        class FakeAgent:  # mimics an Agent (agent_name, id, no swarm_type)
            agent_name = "Quant"
            id = "a1"

        with capture_run("Agent.run", FakeAgent(), task="t") as h:
            h.record_output("o")
        a = _attrs(_by_name(spans, "Agent.run"))
        assert a["swarms.component"] == "FakeAgent"
        assert a["swarms.name"] == "Quant"
        assert a["swarms.id"] == "a1"
        assert "swarms.swarm_type" not in a  # never on a single agent

    def test_swarm_identity_has_swarm_type(self, spans):
        class FakeSwarm:
            name = "Router"
            id = "s1"
            swarm_type = "SequentialWorkflow"

        with capture_run("SwarmRouter.run", FakeSwarm()) as h:
            h.record_output("o")
        a = _attrs(_by_name(spans, "SwarmRouter.run"))
        assert a["swarms.name"] == "Router"
        assert a["swarms.swarm_type"] == "SequentialWorkflow"

    def test_operation_name_agent_vs_swarm(self, spans):
        class Agent:  # class named exactly "Agent"
            agent_name = "A"

        class Workflow:
            name = "W"

        with capture_run("x", Agent()) as h:
            h.record_output("o")
        with capture_run("y", Workflow()) as h:
            h.record_output("o")
        assert (
            _attrs(_by_name(spans, "x"))["gen_ai.operation.name"]
            == "agent"
        )
        assert (
            _attrs(_by_name(spans, "y"))["gen_ai.operation.name"]
            == "swarm"
        )


# ===========================================================================
# capture_init
# ===========================================================================
class TestCaptureInit:
    def test_emits_init_span_with_config(self, spans):
        class Widget:
            def __init__(self, size=3, label="w"):
                self.size = size
                self.label = label
                capture_init(self)

        Widget(size=7)
        span = _by_name(spans, "Widget.init")
        assert span is not None
        cfg = json.loads(_attrs(span)["swarms.config"])
        assert cfg == {"size": 7, "label": "w"}


# ===========================================================================
# log_agent_data (OTel replacement for the old swarms.world POST)
# ===========================================================================
class TestLogAgentData:
    def test_emits_state_span(self, spans):
        log_agent_data(
            {"agent_name": "Q", "id": "a1", "max_loops": 3}
        )
        span = _by_name(spans, "swarms.state")
        a = _attrs(span)
        assert a["swarms.name"] == "Q"
        assert a["swarms.id"] == "a1"
        assert "swarms.state" in a

    def test_reexported_from_package(self):
        from swarms.telemetry import log_agent_data as reexport

        assert reexport is log_agent_data

    def test_old_impl_deleted(self):
        import swarms.telemetry.main as main

        assert not hasattr(main, "log_agent_data")
        assert not hasattr(main, "_log_agent_data")


# ===========================================================================
# Disabled path — everything is a safe no-op
# ===========================================================================
class TestDisabledPath:
    def test_disabled_instance_is_inert(self, monkeypatch):
        monkeypatch.setenv("SWARMS_TELEMETRY_ON", "false")
        telem = SwarmTelemetry()  # fresh instance, not the singleton
        assert telem.ready is False
        with telem.capture_run("X.run", None, task="hi") as h:
            h.record_output("y")
            h.record_error(ValueError("z"))  # no raise
        telem.capture_init(object())
        telem.capture_error(ValueError("z"), None)  # all no-ops


# ===========================================================================
# Integration — instrumentation on real structures
# ===========================================================================
def _load_arch_classes():
    from swarms import (
        Agent,
        AgentRearrange,
        ConcurrentWorkflow,
        CouncilAsAJudge,
        GraphWorkflow,
        GroupChat,
        HeavySwarm,
        HierarchicalSwarm,
        MajorityVoting,
        MixtureOfAgents,
        MultiAgentRouter,
        RoundRobinSwarm,
        SequentialWorkflow,
        SwarmRouter,
    )
    from swarms.structs.batched_grid_workflow import (
        BatchedGridWorkflow,
    )
    from swarms.structs.debate_with_judge import DebateWithJudge
    from swarms.structs.llm_council import LLMCouncil
    from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm

    return {
        "Agent": Agent,
        "SwarmRouter": SwarmRouter,
        "ConcurrentWorkflow": ConcurrentWorkflow,
        "SequentialWorkflow": SequentialWorkflow,
        "HierarchicalSwarm": HierarchicalSwarm,
        "MixtureOfAgents": MixtureOfAgents,
        "MajorityVoting": MajorityVoting,
        "GroupChat": GroupChat,
        "HeavySwarm": HeavySwarm,
        "RoundRobinSwarm": RoundRobinSwarm,
        "AgentRearrange": AgentRearrange,
        "GraphWorkflow": GraphWorkflow,
        "MultiAgentRouter": MultiAgentRouter,
        "CouncilAsAJudge": CouncilAsAJudge,
        "DebateWithJudge": DebateWithJudge,
        "PlannerWorkerSwarm": PlannerWorkerSwarm,
        "BatchedGridWorkflow": BatchedGridWorkflow,
        "LLMCouncil": LLMCouncil,
    }


ARCH = _load_arch_classes()


class TestInstrumentationCoverage:
    @pytest.mark.parametrize("name", sorted(ARCH))
    def test_run_is_traced(self, name):
        import inspect

        cls = ARCH[name]
        run = cls.run
        # A run() is instrumented either via the @trace_run decorator
        # (sets __wrapped__) or via an inline `capture_run(...)` block
        # (SwarmRouter). Accept either.
        decorated = hasattr(run, "__wrapped__")
        inline = "capture_run(" in inspect.getsource(run)
        assert (
            decorated or inline
        ), f"{name}.run is not instrumented (no @trace_run and no capture_run block)"

    @pytest.mark.parametrize("name", sorted(ARCH))
    def test_init_calls_capture_init(self, name):
        import inspect

        src = inspect.getsource(ARCH[name].__init__)
        assert (
            "capture_init(self)" in src
        ), f"{name}.__init__ does not call capture_init(self)"


class TestConstructionEmitsInitSpan:
    """Construct the cheaply-constructible structures and assert init spans."""

    def test_agent(self, spans):
        _agent()
        assert _by_name(spans, "Agent.init") is not None

    def test_swarm_router(self, spans):
        from swarms import SwarmRouter

        SwarmRouter(name="R", agents=[_agent()])
        span = _by_name(spans, "SwarmRouter.init")
        assert span is not None
        # config carries the constructor params, not internal cache state
        cfg = json.loads(_attrs(span)["swarms.config"])
        assert "swarm_type" in cfg
        assert "_swarm_cache" not in cfg
        assert _attrs(span)["swarms.swarm_type"]

    def test_concurrent_workflow(self, spans):
        from swarms import ConcurrentWorkflow

        ConcurrentWorkflow(agents=[_agent()])
        assert _by_name(spans, "ConcurrentWorkflow.init") is not None

    def test_sequential_workflow(self, spans):
        from swarms import SequentialWorkflow

        SequentialWorkflow(agents=[_agent()])
        assert _by_name(spans, "SequentialWorkflow.init") is not None

    def test_majority_voting(self, spans):
        from swarms import MajorityVoting

        MajorityVoting(agents=[_agent()])
        assert _by_name(spans, "MajorityVoting.init") is not None


# ===========================================================================
# Real LLM runs — actually execute agents/swarms and assert the run is logged.
# Gated on a live API key; makes real (cheap) LLM calls with gpt-4o-mini.
# ===========================================================================
@requires_llm
class TestRealLLMRuns:
    def test_agent_run_is_logged(self, spans):
        agent = _agent("RealAgent")
        result = agent.run(
            task="Reply with exactly the word PONG and nothing else."
        )
        assert (
            result and str(result).strip()
        ), "agent returned empty output"

        span = _by_name(spans, "Agent.run")
        assert span is not None, "Agent.run span was not logged"
        a = _attrs(span)
        assert a["swarms.component"] == "Agent"
        assert a["swarms.name"] == "RealAgent"
        assert a["swarms.status"] == "completed"
        assert a["swarms.input.task"].startswith("Reply with")
        assert a[
            "swarms.output"
        ].strip(), "run output not logged on the span"
        assert (
            a["gen_ai.operation.name"] == "agent"
        )  # single agent, not a swarm

    def test_sequential_workflow_run_is_logged(self, spans):
        from swarms import SequentialWorkflow

        wf = SequentialWorkflow(
            agents=[_agent("Seq-1"), _agent("Seq-2")]
        )
        result = wf.run(task="Say hello in one short sentence.")
        assert result

        parent = _by_name(spans, "SequentialWorkflow.run")
        assert (
            parent is not None
        ), "SequentialWorkflow.run span not logged"
        assert _attrs(parent)["swarms.status"] == "completed"
        # the child agents' runs are logged too
        agent_runs = [
            s
            for s in spans.get_finished_spans()
            if s.name == "Agent.run"
        ]
        assert (
            len(agent_runs) >= 1
        ), "child Agent.run spans were not logged"

    def test_concurrent_workflow_run_is_logged(self, spans):
        from swarms import ConcurrentWorkflow

        wf = ConcurrentWorkflow(
            agents=[_agent("Conc-1"), _agent("Conc-2")]
        )
        wf.run(task="Name one primary color.")

        span = _by_name(spans, "ConcurrentWorkflow.run")
        assert (
            span is not None
        ), "ConcurrentWorkflow.run span not logged"
        a = _attrs(span)
        assert a["swarms.status"] == "completed"
        assert a["gen_ai.operation.name"] == "swarm"
        # both agents actually ran and were logged
        agent_runs = [
            s
            for s in spans.get_finished_spans()
            if s.name == "Agent.run"
        ]
        assert (
            len(agent_runs) >= 1
        ), "child Agent.run spans not logged"

    def test_swarm_router_run_is_logged(self, spans):
        from swarms import SwarmRouter

        router = SwarmRouter(
            name="RealRouter",
            agents=[_agent("Router-1")],
            swarm_type="SequentialWorkflow",
        )
        result = router.run(task="Reply with the word OK.")
        assert result

        span = _by_name(spans, "SwarmRouter.run")
        assert span is not None, "SwarmRouter.run span not logged"
        a = _attrs(span)
        assert a["swarms.status"] == "completed"
        assert a["swarms.swarm_type"] == "SequentialWorkflow"
        assert a["swarms.output"].strip()

    def test_agent_run_error_is_logged(self, spans):
        """A real run against a bad model logs an error span.

        The Agent's retry loop swallows LLM errors (it does not re-raise), so
        the error is captured via ``capture_error`` as an ``Agent.llm_error``
        span rather than as a raised exception on ``Agent.run``.
        """
        from swarms import Agent

        agent = Agent(
            agent_name="BadModelAgent",
            model_name="nonexistent-model-swarms-otel-test",
            max_loops=1,
            retry_attempts=1,
            persistent_memory=False,
            print_on=False,
            verbose=False,
        )
        # Does not necessarily raise — the agent swallows the LLM error.
        try:
            agent.run(task="hi")
        except Exception:
            pass

        err = _by_name(spans, "Agent.llm_error")
        assert (
            err is not None
        ), "Agent.llm_error span was not logged for a failed LLM call"
        a = _attrs(err)
        assert a["swarms.status"] == "error"
        assert a["swarms.component"] == "Agent"
        assert "swarms.error.type" in a
        assert any(e.name == "exception" for e in err.events)


if __name__ == "__main__":
    pytest.main()
