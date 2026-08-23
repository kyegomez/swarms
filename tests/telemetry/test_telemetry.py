"""
OpenTelemetry coverage for swarms, end to end.

One file, four concerns that used to live in four:

* **Primitives** - ``capture_run``, ``capture_error``, ``trace_run``,
  ``capture_init``, the payload schema, and the disabled path.
* **Invariants** - properties that must hold for every architecture: span
  parentage, telemetry off by default, behaviour never changed by telemetry,
  a broken backend never breaking a run, payload bounds, attribute-schema
  consistency, no duplicate or leaked spans.
* **Core architectures** - SequentialWorkflow, ConcurrentWorkflow,
  AgentRearrange, RoundRobinSwarm, MixtureOfAgents, MajorityVoting,
  BatchedGridWorkflow, SwarmRouter.
* **Advanced architectures** - HierarchicalSwarm, MultiAgentRouter,
  GraphWorkflow, GroupChat, DebateWithJudge, CouncilAsAJudge, LLMCouncil,
  HeavySwarm, PlannerWorkerSwarm.
* **Identity helpers** - ``generate_user_id`` and ``get_machine_id``.

Everything here is offline. No test needs an API key or a network route: every
LLM call is served by :class:`FakeLLM`, and the telemetry base URL is pointed
at a dead local address before the provider is built, so no span ever leaves
the machine. The one exception is :class:`TestRealLLMRuns`, which is skipped
unless a provider key is present.

The four source files each carried their own copy of the exporter fixture, the
span helpers, ``FakeLLM`` and ``fake_agent``. Those are defined once here.

Run:
    cd /Users/swarms_wd/Desktop/research/swarms
    PYTHONPATH=. python3 -m pytest tests/telemetry/test_telemetry.py -q -p no:randomly
"""

import json
import os
import time
import uuid
from contextlib import contextmanager

import pytest
from dotenv import load_dotenv

load_dotenv()

# Telemetry is off unless asked for. Set before importing the module so the
# gate is read correctly on first import.
os.environ["SWARMS_TELEMETRY_ON"] = "true"

import swarms.telemetry.otel as otel  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from swarms import (  # noqa: E402
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
    RESPOND_TOOL,
    RoundRobinSwarm,
    SequentialWorkflow,
    SwarmRouter,
)
from swarms.schemas.planner_worker_schemas import (  # noqa: E402
    CycleVerdict,
)
from swarms.structs.batched_grid_workflow import (  # noqa: E402
    BatchedGridWorkflow,
)
from swarms.structs.council_as_judge import (  # noqa: E402
    EvaluationError,
)
from swarms.structs.debate_with_judge import (  # noqa: E402
    DebateWithJudge,
)
from swarms.structs.llm_council import LLMCouncil  # noqa: E402
from swarms.structs.planner_worker_swarm import (  # noqa: E402
    PlannerWorkerSwarm,
)
from swarms.telemetry.main import (  # noqa: E402
    generate_user_id,
    get_machine_id,
)
from swarms.telemetry.otel import (  # noqa: E402
    MAX_CONFIG_CHARS,
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

LLM_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_llm_keys(request):
    """
    Strip provider keys so an offline test cannot silently reach a real LLM.

    Skipped for the live-run tests, which are gated on a key being present.
    """
    if request.node.get_closest_marker("requires_llm_key"):
        yield
        return

    saved = {k: os.environ.pop(k, None) for k in LLM_KEY_VARS}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# The live tests build an Agent on gpt-4o-mini, so gate on that provider's
# key specifically. Gating on "any provider key" let them run with only a
# Groq or Gemini key present and fail on an empty OpenAI credential.
requires_llm = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — real-run tests skipped",
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def _exporter():
    """
    A telemetry provider whose spans are captured in memory.

    The receiver URL is a module constant, so it is pointed at a dead local
    address *before* the provider is built and the ``lru_cache``d singleton is
    rebuilt so the env gate is re-read. Nothing emitted by this file leaves the
    machine.
    """
    os.environ["SWARMS_TELEMETRY_ON"] = "true"
    otel.TELEMETRY_BASE_URL = "http://127.0.0.1:9/dead"
    otel.swarm_telemetry.cache_clear()

    telem = otel.swarm_telemetry()
    assert (
        telem.ready
    ), "telemetry must be ready with SWARMS_TELEMETRY_ON=true"

    mem = InMemorySpanExporter()
    telem._provider.add_span_processor(SimpleSpanProcessor(mem))
    return mem


@pytest.fixture
def spans(_exporter):
    """The exporter, cleared so each test sees only its own spans."""
    _exporter.clear()
    return _exporter


@pytest.fixture
def toggle_telemetry(_exporter):
    """Context manager fixture to force telemetry on/off for a block.

    Restores the previous env var, clears the lru_cache singleton, and
    re-attaches ``_exporter`` to the restored (on) instance on exit — so
    later tests in this file (and other files in the same session, which
    build their own singleton on first use) are unaffected.
    """

    @contextmanager
    def _toggle(on: bool):
        prev = os.environ.get("SWARMS_TELEMETRY_ON")
        # Telemetry is opt-out, so "off" must be set explicitly — unsetting the
        # variable would enable it.
        os.environ["SWARMS_TELEMETRY_ON"] = "true" if on else "false"
        otel.swarm_telemetry.cache_clear()
        telem = otel.swarm_telemetry()
        if on:
            telem._provider.add_span_processor(
                SimpleSpanProcessor(_exporter)
            )
        try:
            yield telem
        finally:
            if prev is None:
                os.environ.pop("SWARMS_TELEMETRY_ON", None)
            else:
                os.environ["SWARMS_TELEMETRY_ON"] = prev
            otel.swarm_telemetry.cache_clear()
            restored = otel.swarm_telemetry()
            assert (
                restored.ready
            ), "failed to restore telemetry singleton"
            restored._provider.add_span_processor(
                SimpleSpanProcessor(_exporter)
            )

    return _toggle


# ============================================================================
# Span helpers
# ============================================================================


def _finished(exporter):
    return exporter.get_finished_spans()


def _by_name(exporter, name):
    """The first finished span with this name, or None."""
    for span in _finished(exporter):
        if span.name == name:
            return span
    return None


def _all_by_name(exporter, name):
    """Every finished span with this name."""
    return [s for s in _finished(exporter) if s.name == name]


# Alias: the invariants suite used this spelling.
_by_name_all = _all_by_name


def _attrs(span):
    return dict(span.attributes)


# ============================================================================
# Fake LLM and agent factories
# ============================================================================


class FakeLLM:
    """
    Stand-in for LiteLLM: no network, deterministic reply.

    ``reply`` can be any object (str, dict, list) — ``Agent`` forwards it
    verbatim (see ``LLMManager.call``'s non-streaming path), so callers pick
    whatever shape the consuming orchestrator's parser expects. Set
    ``raise_exc`` to make the call blow up instead.
    """

    def __init__(self, reply="FAKE OUTPUT", raise_exc=False):
        self.stream = False
        self.temperature = 0.5
        self.reply = reply
        self.raise_exc = raise_exc

    def run(self, task=None, img=None, **kwargs):
        if self.raise_exc:
            raise RuntimeError("synthetic LLM failure")
        return self.reply


def fake_agent(
    name, reply=None, raise_exc=False, output_type=None, **kwargs
):
    """
    A real ``Agent`` wired to a ``FakeLLM``.

    Args:
        output_type: Left unset by default, so the agent behaves like any
            other. Pass ``"final"`` when the test needs ``agent.run()`` to
            return the raw reply object unchanged — architectures that parse a
            structured (list/dict) reply depend on that.
    """
    settings = dict(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        verbose=False,
    )
    if output_type is not None:
        settings["output_type"] = output_type
    settings.update(kwargs)

    agent = Agent(**settings)
    agent.llm = FakeLLM(
        reply if reply is not None else f"{name} output",
        raise_exc=raise_exc,
    )
    return agent


def fake_agent_final(name, reply=None, **kwargs):
    """``fake_agent`` with ``output_type="final"``, for structured replies."""
    return fake_agent(
        name, reply=reply, output_type="final", **kwargs
    )


def break_agent(agent, message="synthetic agent failure"):
    """
    Make ``agent.run(...)`` raise directly.

    A raising ``FakeLLM`` is not enough: ``Agent._run`` catches everything the
    LLM raises, retries, and then returns normally without re-raising, so the
    exception never reaches the surrounding swarm. Replacing the bound method
    outright is the only way to observe how each *structure* handles a member
    that is completely broken. It also means no ``Agent.run`` span is emitted
    for that agent, since ``@trace_run`` lives on the method just shadowed.
    """

    def _raise(*args, **kwargs):
        raise RuntimeError(message)

    agent.run = _raise
    return agent


def break_member_run(agent, message="member agent blew up"):
    """``break_agent`` under the name the core suite used, with its message."""
    return break_agent(agent, message=message)


def _agent(name="TelemetryTestAgent"):
    """A real agent for the live-LLM tests (no FakeLLM)."""
    return Agent(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        verbose=False,
    )


# ==========================================================================
# Primitives: capture_run, capture_error, trace_run, capture_init, schema
# ==========================================================================


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
@pytest.mark.requires_llm_key
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


# ==========================================================================
# Invariants that must hold for every architecture
# ==========================================================================


class TestSpanParentageInvariant:
    """capture_run activates its span as current (via trace.use_span) and
    thread pools carry the context across worker threads, so a swarm run and
    every agent run beneath it share one trace and form a parent/child tree a
    viewer can reconstruct. Spans emitted outside a run — the ``*.init`` spans
    from construction — remain independent roots.
    """

    def test_two_agent_sequential_workflow_nests_into_one_trace(
        self, spans
    ):
        from swarms import SequentialWorkflow

        a1, a2 = fake_agent("Flat-1"), fake_agent("Flat-2")
        wf = SequentialWorkflow(agents=[a1, a2], max_loops=1)
        result = wf.run("hello")
        assert result

        finished = _finished(spans)
        # Agent.init x2, AgentRearrange.init, SequentialWorkflow.init,
        # Agent.run x2, AgentRearrange.run, SequentialWorkflow.run.
        assert len(finished) == 8

        init_spans = [s for s in finished if s.name.endswith(".init")]
        run_spans = [s for s in finished if s.name.endswith(".run")]
        assert len(init_spans) == 4
        assert len(run_spans) == 4

        # init spans happen at construction, outside any run: still roots.
        for s in init_spans:
            assert (
                s.parent is None
            ), f"{s.name} unexpectedly has a parent"

        # Every run span shares one trace.
        run_trace_ids = {
            s.get_span_context().trace_id for s in run_spans
        }
        assert (
            len(run_trace_ids) == 1
        ), f"run spans split across {len(run_trace_ids)} traces — nesting broke"

        # SequentialWorkflow.run is the root of that trace.
        swarm_run = _by_name_all(spans, "SequentialWorkflow.run")[0]
        assert swarm_run.parent is None

        # SequentialWorkflow delegates to AgentRearrange, which runs the
        # agents — so the real tree is workflow -> rearrange -> agents.
        rearrange_run = _by_name_all(spans, "AgentRearrange.run")[0]
        assert (
            rearrange_run.parent is not None
            and rearrange_run.parent.span_id
            == swarm_run.get_span_context().span_id
        ), "AgentRearrange.run is not a child of SequentialWorkflow.run"

        agent_runs = _by_name_all(spans, "Agent.run")
        assert len(agent_runs) == 2
        for ar in agent_runs:
            assert (
                ar.parent is not None
                and ar.parent.span_id
                == rearrange_run.get_span_context().span_id
            ), "Agent.run is not a child of AgentRearrange.run"

    def test_concurrent_workflow_nests_across_worker_threads(
        self, spans
    ):
        """OTel context does not cross threads on its own; the pools use
        ContextThreadPoolExecutor so concurrent agents still nest."""
        from swarms import ConcurrentWorkflow

        wf = ConcurrentWorkflow(
            agents=[fake_agent("Conc-1"), fake_agent("Conc-2")]
        )
        wf.run("hello")

        run_spans = [
            s for s in _finished(spans) if s.name.endswith(".run")
        ]
        assert (
            len({s.get_span_context().trace_id for s in run_spans})
            == 1
        ), "concurrent agent spans orphaned into separate traces"

        parent = _by_name_all(spans, "ConcurrentWorkflow.run")[0]
        for ar in _by_name_all(spans, "Agent.run"):
            assert (
                ar.parent is not None
                and ar.parent.span_id
                == parent.get_span_context().span_id
            ), "Agent.run in a worker thread is not a child of the workflow run"


# ===========================================================================
# 2. Telemetry is off by default
# ===========================================================================
class TestTelemetryOffByDefault:
    def test_full_run_produces_no_spans_when_unset(
        self, toggle_telemetry, _exporter
    ):
        with toggle_telemetry(on=False) as telem:
            assert telem.ready is False
            _exporter.clear()

            from swarms import SequentialWorkflow

            a1, a2 = fake_agent("Off-1"), fake_agent("Off-2")
            wf = SequentialWorkflow(agents=[a1, a2], max_loops=1)
            result = wf.run("hello, are you off?")

            # The run still works correctly — telemetry being off never
            # breaks functionality.
            assert isinstance(result, list)
            assert any(
                "Off-1 output" in str(m.get("content", ""))
                for m in result
            )
            assert any(
                "Off-2 output" in str(m.get("content", ""))
                for m in result
            )

            assert list(_exporter.get_finished_spans()) == []


# ===========================================================================
# 3. Telemetry never changes program behavior
# ===========================================================================
class TestTelemetryNeverChangesBehavior:
    def test_agent_output_identical_on_vs_off(
        self, toggle_telemetry, spans
    ):
        on_agent = fake_agent("Parity-On", reply="the same output")
        on_result = on_agent.run(task="hi")

        with toggle_telemetry(on=False):
            off_agent = fake_agent(
                "Parity-Off", reply="the same output"
            )
            off_result = off_agent.run(task="hi")

        assert on_result == off_result == "the same output"

    def test_swarm_output_identical_on_vs_off(
        self, toggle_telemetry, spans
    ):
        from swarms import SequentialWorkflow

        def build():
            return SequentialWorkflow(
                agents=[
                    fake_agent("Par-1", reply="r1"),
                    fake_agent("Par-2", reply="r2"),
                ],
                max_loops=1,
            )

        on_result = build().run("same task")
        with toggle_telemetry(on=False):
            off_result = build().run("same task")

        assert on_result == off_result

    def test_exception_propagates_identically_on_vs_off(
        self, toggle_telemetry, spans
    ):
        class Boom:
            @trace_run("Boom.run")
            def run(self, task=None):
                raise ValueError("distinctive-boom-message-42")

        with pytest.raises(ValueError) as exc_on:
            Boom().run(task="x")

        with toggle_telemetry(on=False):
            with pytest.raises(ValueError) as exc_off:
                Boom().run(task="x")

        assert type(exc_on.value) is type(exc_off.value)
        assert str(exc_on.value) == str(exc_off.value)

    def test_trace_run_preserves_function_metadata(self, spans):
        class Comp:
            @trace_run("Comp.documented", input_params=("task",))
            def run(self, task=None):
                """Original docstring."""
                return task

        assert Comp.run.__name__ == "run"
        assert Comp.run.__doc__ == "Original docstring."
        assert hasattr(Comp.run, "__wrapped__")

        import inspect

        sig = inspect.signature(Comp.run.__wrapped__)
        assert list(sig.parameters) == ["self", "task"]

    def test_non_string_return_values_pass_through_unchanged(
        self, toggle_telemetry, spans
    ):
        payload = {"a": [1, 2, 3], "b": None, "c": {"nested": True}}

        class Ret:
            @trace_run("Ret.run")
            def run(self, task=None):
                return payload

        on_result = Ret().run(task="x")
        with toggle_telemetry(on=False):
            off_result = Ret().run(task="x")

        assert on_result == payload
        assert off_result == payload
        assert on_result == off_result

        list_payload = [1, "two", 3.0, None]

        class RetList:
            @trace_run("RetList.run")
            def run(self, task=None):
                return list_payload

        assert RetList().run(task="x") == list_payload


# ===========================================================================
# 4. A broken exporter/tracer/span never breaks a run
# ===========================================================================
class TestBrokenBackendNeverBreaksRun:
    def test_exporter_export_raises_does_not_break_run(self, spans):
        class BadExporter:
            def export(self, spans_):
                raise RuntimeError("export exploded")

            def shutdown(self):
                pass

            def force_flush(self, timeout_millis=30000):
                return True

        telem = otel.swarm_telemetry()
        telem._provider.add_span_processor(
            SimpleSpanProcessor(BadExporter())
        )

        # Must not raise, and the good in-memory exporter still gets the
        # span (SimpleSpanProcessor isolates exporter failures per
        # processor).
        with capture_run("BadExport.run", None) as h:
            h.record_output("still fine")
        assert _by_name_all(spans, "BadExport.run")

    def test_tracer_start_span_raises_does_not_break_run(
        self, spans, monkeypatch
    ):
        telem = otel.swarm_telemetry()

        def boom(*args, **kwargs):
            raise RuntimeError("tracer exploded")

        monkeypatch.setattr(telem._tracer, "start_span", boom)

        # capture_run swallows the failure and yields a no-op handle.
        with capture_run("BrokenTracer.run", None, task="x") as h:
            h.record_output("survived")
            h.record_error(ValueError("also survived"))

        # No span with this name was recorded (start_span never succeeded),
        # but nothing raised into the caller.
        assert _by_name_all(spans, "BrokenTracer.run") == []

    def test_span_set_attribute_raises_does_not_break_run(
        self, spans, monkeypatch
    ):
        telem = otel.swarm_telemetry()
        real_start_span = telem._tracer.start_span

        def wrapped(*args, **kwargs):
            span = real_start_span(*args, **kwargs)

            def boom_attr(key, value):
                raise RuntimeError("set_attribute exploded")

            monkeypatch.setattr(span, "set_attribute", boom_attr)
            return span

        monkeypatch.setattr(telem._tracer, "start_span", wrapped)

        # No obj/inputs, so span setup itself never calls set_attribute —
        # only record_output does, exercising _SpanHandle's own try/except.
        with capture_run("BrokenAttr.run", None) as h:
            h.record_output("output survives a raising attribute")

    def test_agent_run_survives_a_hostile_backend(self, spans):
        """End-to-end: a full agent run completes even with a hostile
        exporter attached alongside the (also unreachable) real one.
        """

        class HostileExporter:
            def export(self, spans_):
                raise RuntimeError("hostile export")

            def shutdown(self):
                # Deliberately a no-op (not raising): this processor is
                # attached to the shared, process-wide provider for the rest
                # of the test session, and a raising shutdown() would blow up
                # the interpreter's atexit handler for every other test file.
                # The invariant under test is export-time robustness, not
                # shutdown-time.
                pass

            def force_flush(self, timeout_millis=30000):
                return True

        telem = otel.swarm_telemetry()
        telem._provider.add_span_processor(
            SimpleSpanProcessor(HostileExporter())
        )

        agent = fake_agent("Hostile-Survivor", reply="i survived")
        result = agent.run(task="hi")
        assert result == "i survived"

    def test_unreachable_endpoint_never_raises_and_completes_quickly(
        self, spans
    ):
        from swarms import SequentialWorkflow

        start = time.monotonic()
        a1, a2 = fake_agent("Timed-1"), fake_agent("Timed-2")
        wf = SequentialWorkflow(agents=[a1, a2], max_loops=1)
        result = wf.run("hello, dead endpoint")
        elapsed = time.monotonic() - start

        assert result
        assert (
            elapsed < 5.0
        ), f"run against a dead OTLP endpoint took {elapsed:.2f}s — telemetry is blocking"


# ===========================================================================
# 5. Payload bounds and config serialization
# ===========================================================================
class TestPayloadBoundsAndConfig:
    def test_huge_output_and_input_task_are_truncated(self, spans):
        huge = "z" * (MAX_PAYLOAD_CHARS + 500)
        agent = fake_agent("Huge-Output", reply=huge)
        agent.run(task=huge)

        run_span = _by_name_all(spans, "Agent.run")[-1]
        a = _attrs(run_span)

        assert a["swarms.output"].endswith("…[truncated]")
        assert len(a["swarms.output"]) <= MAX_PAYLOAD_CHARS + len(
            "…[truncated]"
        )
        assert a["swarms.input.task"].endswith("…[truncated]")
        assert len(a["swarms.input.task"]) <= MAX_PAYLOAD_CHARS + len(
            "…[truncated]"
        )

    def test_huge_config_bounded_by_max_config_chars(self):
        class Huge:
            def __init__(self, blob=None):
                self.blob = blob

        big = "y" * (MAX_CONFIG_CHARS + 1000)
        cfg = init_config(Huge(blob=big))
        assert cfg.endswith("…[truncated]")
        assert len(cfg) <= MAX_CONFIG_CHARS + len("…[truncated]")

    def test_real_multi_agent_swarm_config_size(self, spans):
        """Report the real swarms.config size for a multi-agent swarm."""
        from swarms import SwarmRouter

        agents = [fake_agent(f"CfgSize-{i}") for i in range(3)]
        SwarmRouter(
            name="CfgSizeRouter",
            agents=agents,
            swarm_type="ConcurrentWorkflow",
        )
        span = _by_name_all(spans, "SwarmRouter.init")[-1]
        cfg = _attrs(span)["swarms.config"]
        assert len(cfg) <= MAX_CONFIG_CHARS + len("…[truncated]")
        assert 0 < len(cfg) < MAX_CONFIG_CHARS
        # Surface the number for the human report (visible with `-s`).
        print(
            f"\nswarms.config size (3-agent SwarmRouter): {len(cfg)} bytes"
        )

    # -- init_config hardening --------------------------------------------

    def test_never_raises_bad_repr_degrades_siblings_survive(self):
        class BadRepr:
            def __repr__(self):
                raise RuntimeError("repr boom")

            def __str__(self):
                raise RuntimeError("str boom")

        class Holder:
            def __init__(self, good=1, bad=None):
                self.good = good
                self.bad = bad

        cfg = json.loads(init_config(Holder(good=5, bad=BadRepr())))
        assert cfg["good"] == 5
        assert cfg["bad"] == "<unserializable BadRepr>"

    def test_never_raises_self_referential_container_degrades(self):
        cyclic = []
        cyclic.append(cyclic)

        class Holder:
            def __init__(self, good=1, bad=None):
                self.good = good
                self.bad = bad

        cfg = json.loads(init_config(Holder(good=9, bad=cyclic)))
        assert cfg["good"] == 9
        assert cfg["bad"] == "<unserializable list>"

    def test_functions_classes_and_partial_render_without_address(
        self,
    ):
        import functools

        def my_tool(x):
            """A tool."""

        class SomeClass:
            pass

        class Holder:
            def __init__(self, fn=None, cls=None, part=None):
                self.fn = fn
                self.cls = cls
                self.part = part

        cfg = json.loads(
            init_config(
                Holder(
                    fn=my_tool,
                    cls=SomeClass,
                    part=functools.partial(my_tool, 1),
                )
            )
        )
        assert cfg["fn"].endswith(".my_tool")
        assert "0x" not in cfg["fn"]
        assert cfg["cls"].endswith(".SomeClass")
        assert "0x" not in cfg["cls"]
        assert "my_tool" in cfg["part"]
        assert "0x" not in cfg["part"]

    def test_agent_config_excludes_internal_managers(self):
        """A real Agent's config never contains llm_manager/mcp_manager —
        they're runtime internals, not __init__ parameters, even though
        Agent.to_dict() exposes them."""
        agent = fake_agent("ConfigCheck")
        cfg = json.loads(init_config(agent))
        assert "llm_manager" not in cfg
        assert "mcp_manager" not in cfg

    def test_swarm_router_config_excludes_cache_and_factory(self):
        from swarms import SwarmRouter

        router = SwarmRouter(
            name="ExclCheck", agents=[fake_agent("ExclAgent")]
        )
        cfg = json.loads(init_config(router))
        assert "_swarm_cache" not in cfg
        assert "_swarm_factory" not in cfg
        assert (
            "swarm_type" in cfg
        )  # real constructor param still present

    def test_to_dict_raising_is_ignored_not_fatal(self):
        class RaisingToDict:
            def __init__(self, x=1):
                self.x = x

            def to_dict(self):
                raise RuntimeError("to_dict boom")

        cfg = json.loads(init_config(RaisingToDict(x=3)))
        assert cfg == {"x": 3}

    def test_to_dict_requiring_args_is_ignored(self):
        class RequiresArgToDict:
            def __init__(self, x=1):
                self.x = x

            def to_dict(self, extra):
                return {"x": "should never be used"}

        cfg = json.loads(init_config(RequiresArgToDict(x=4)))
        assert cfg == {"x": 4}

    def test_to_dict_returning_non_dict_is_ignored(self):
        class NonDictToDict:
            def __init__(self, x=1):
                self.x = x

            def to_dict(self):
                return "not a dict"

        cfg = json.loads(init_config(NonDictToDict(x=5)))
        assert cfg == {"x": 5}

    def test_no_str_defaults_to_classname_paren_name(self):
        agents = [fake_agent("A"), fake_agent("B")]

        class Holder:
            def __init__(self, items=None):
                self.items = items

        cfg = json.loads(init_config(Holder(items=agents)))
        assert cfg["items"] == ["Agent(A)", "Agent(B)"]
        for entry in cfg["items"]:
            assert "0x" not in entry

    def test_objects_with_str_keep_their_str(self):
        class WithStr:
            def __str__(self):
                return "custom-str-value"

        class Holder:
            def __init__(self, thing=None):
                self.thing = thing

        cfg = json.loads(init_config(Holder(thing=WithStr())))
        assert cfg["thing"] == "custom-str-value"

    def test_no_config_anywhere_contains_a_memory_address(self):
        agent = fake_agent("AddressCheck")
        cfg = init_config(agent)
        assert "0x" not in cfg

        from swarms import SwarmRouter

        router = SwarmRouter(
            name="AddressCheckRouter", agents=[fake_agent("ACR-1")]
        )
        router_cfg = init_config(router)
        assert "0x" not in router_cfg

    def test_configs_are_byte_identical_across_identical_constructions(
        self,
    ):
        class Simple:
            def __init__(self, a=1, b="x", items=None):
                self.a = a
                self.b = b
                self.items = items or [1, 2, 3]

        cfg1 = init_config(Simple(a=1, b="x"))
        cfg2 = init_config(Simple(a=1, b="x"))
        assert cfg1 == cfg2

        # Two independently-constructed Simple() instances with identical
        # ctor args serialize byte-for-byte identically.
        cfg3 = init_config(Simple(a=1, b="x"))
        assert cfg1 == cfg3

        # For a real Agent, re-serializing the SAME instance is byte-for-byte
        # deterministic (no run-to-run drift from dict ordering, etc). Two
        # independently-constructed Agents are deliberately NOT compared here
        # — Agent auto-assigns a fresh random `id` per instance, so their
        # configs legitimately differ even with identical explicit kwargs.
        agent = fake_agent("ByteIdentical")
        agent_cfg_1 = init_config(agent)
        agent_cfg_2 = init_config(agent)
        assert agent_cfg_1 == agent_cfg_2


# ===========================================================================
# 6. Attribute schema consistency
# ===========================================================================
class TestAttributeSchemaConsistency:
    REQUIRED_KEYS = (
        "swarms.component",
        "swarms.name",
        "swarms.status",
        "gen_ai.operation.name",
    )

    def test_agent_run_span_has_full_schema_and_operation_agent(
        self, spans
    ):
        agent = fake_agent("SchemaAgent")
        agent.run(task="hi")
        span = _by_name_all(spans, "Agent.run")[-1]
        a = _attrs(span)
        for key in self.REQUIRED_KEYS:
            assert key in a, f"missing {key} on Agent.run"
        assert a["gen_ai.operation.name"] == "agent"
        assert "swarms.swarm_type" not in a

    def test_multi_agent_structures_have_full_schema_and_operation_swarm(
        self, spans
    ):
        from swarms import ConcurrentWorkflow, SequentialWorkflow

        cases = [
            (
                "ConcurrentWorkflow",
                ConcurrentWorkflow(
                    agents=[
                        fake_agent("SchemaConc-1"),
                        fake_agent("SchemaConc-2"),
                    ]
                ),
            ),
            (
                "SequentialWorkflow",
                SequentialWorkflow(
                    agents=[
                        fake_agent("SchemaSeq-1"),
                        fake_agent("SchemaSeq-2"),
                    ],
                    max_loops=1,
                ),
            ),
        ]
        for cls_name, instance in cases:
            spans_before = len(_finished(spans))
            instance.run("hello")
            run_spans = [
                s
                for s in _finished(spans)[spans_before:]
                if s.name == f"{cls_name}.run"
            ]
            assert run_spans, f"no {cls_name}.run span emitted"
            a = _attrs(run_spans[-1])
            for key in self.REQUIRED_KEYS:
                assert key in a, f"missing {key} on {cls_name}.run"
            assert a["gen_ai.operation.name"] == "swarm"

    def test_swarm_type_only_on_components_defining_it(self, spans):
        from swarms import ConcurrentWorkflow, SwarmRouter

        router = SwarmRouter(
            name="SchemaSwarmType",
            agents=[fake_agent("SchemaST-Router")],
            swarm_type="ConcurrentWorkflow",
        )
        router.run("hello")
        router_span = _by_name_all(spans, "SwarmRouter.run")[-1]
        assert "swarms.swarm_type" in _attrs(router_span)

        conc = ConcurrentWorkflow(
            agents=[fake_agent("SchemaST-Conc")]
        )
        conc.run("hello")
        conc_span = _by_name_all(spans, "ConcurrentWorkflow.run")[-1]
        # ConcurrentWorkflow has no swarm_type attribute of its own.
        assert not hasattr(conc, "swarm_type")
        assert "swarms.swarm_type" not in _attrs(conc_span)

    def test_no_span_attribute_is_none_or_empty_string(self, spans):
        from swarms import SequentialWorkflow

        spans.clear()
        a1, a2 = fake_agent("NoNull-1"), fake_agent("NoNull-2")
        wf = SequentialWorkflow(agents=[a1, a2], max_loops=1)
        wf.run("a real task")

        for span in _finished(spans):
            for key, value in _attrs(span).items():
                assert (
                    value is not None
                ), f"{span.name} attribute {key!r} is None"
                if isinstance(value, str):
                    assert (
                        value != ""
                    ), f"{span.name} attribute {key!r} is an empty string"


# ===========================================================================
# 7. No duplicate/leaked spans
# ===========================================================================
class TestNoDuplicateOrLeakedSpans:
    def test_single_agent_run_emits_exactly_one_span(self, spans):
        agent = fake_agent("SingleRun")
        agent.run(task="once")
        assert len(_by_name_all(spans, "Agent.run")) == 1

    def test_running_same_swarm_twice_emits_exactly_two_swarm_spans(
        self, spans
    ):
        from swarms import ConcurrentWorkflow

        wf = ConcurrentWorkflow(
            agents=[fake_agent("Dup-1"), fake_agent("Dup-2")]
        )
        wf.run("first")
        wf.run("second")
        assert len(_by_name_all(spans, "ConcurrentWorkflow.run")) == 2

    def test_failed_run_emits_exactly_one_span(self, spans):
        class Failing:
            @trace_run("Failing.run")
            def run(self, task=None):
                raise RuntimeError("single failure")

        with pytest.raises(RuntimeError):
            Failing().run(task="x")

        matches = _by_name_all(spans, "Failing.run")
        assert len(matches) == 1
        assert _attrs(matches[0])["swarms.status"] == "error"

    def test_error_after_output_does_not_double_record(self, spans):
        with capture_run("Idempotent.run", None) as h:
            h.record_output("already done")
            h.record_error(ValueError("too late"))
        span = _by_name_all(spans, "Idempotent.run")[-1]
        a = _attrs(span)
        assert a["swarms.status"] == "completed"
        assert "swarms.error.type" not in a
        # exactly one span, no second one snuck in for the "error"
        assert len(_by_name_all(spans, "Idempotent.run")) == 1

    def test_capture_init_still_emits_span_when_config_degrades(
        self, spans
    ):
        """A field that init_config can't serialize must degrade the field,
        not swallow the whole init span (previously it vanished; fixed with
        an end() in `finally`)."""

        class BadRepr:
            def __repr__(self):
                raise RuntimeError("boom")

            def __str__(self):
                raise RuntimeError("boom")

        class Widget:
            def __init__(self, bad=None):
                self.bad = bad
                from swarms.telemetry.otel import capture_init

                capture_init(self)

        Widget(bad=BadRepr())
        matches = _by_name_all(spans, "Widget.init")
        assert len(matches) == 1, "Widget.init span vanished"
        cfg = json.loads(_attrs(matches[0])["swarms.config"])
        assert cfg["bad"] == "<unserializable BadRepr>"


# ==========================================================================
# Core multi-agent architectures
# ==========================================================================


class TestSequentialWorkflowTelemetry:
    def test_happy_path(self, spans):
        a, b = fake_agent("Seq-A"), fake_agent("Seq-B")
        wf = SequentialWorkflow(
            agents=[a, b], max_loops=1, autosave=False
        )
        result = wf.run(task="summarize the news")
        assert result, "workflow returned empty output"

        run_spans = _all_by_name(spans, "SequentialWorkflow.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "SequentialWorkflow"
        assert a_["swarms.name"]
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "summarize the news"
        assert a_["swarms.output"]

        # Child spans: the internal AgentRearrange delegate, plus both agents.
        assert _by_name(spans, "AgentRearrange.run") is not None
        agent_runs = _all_by_name(spans, "Agent.run")
        assert len(agent_runs) == 2
        assert all(
            s.status.status_code.name == "OK" for s in agent_runs
        )

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        """A raising FakeLLM never reaches SequentialWorkflow at all."""
        a, b = fake_agent("Seq-Ok"), fake_agent(
            "Seq-Bad", raise_exc=True
        )
        wf = SequentialWorkflow(
            agents=[a, b], max_loops=1, autosave=False
        )
        result = wf.run(task="hello")  # does not raise
        assert result is not None

        top = _by_name(spans, "SequentialWorkflow.run")
        assert top is not None
        assert _attrs(top)["swarms.status"] == "completed"
        assert top.status.status_code.name == "OK"

        # The LLM failure is only visible as a distinct Agent.llm_error span.
        err = _by_name(spans, "Agent.llm_error")
        assert err is not None
        assert _attrs(err)["swarms.status"] == "error"
        assert _attrs(err)["swarms.name"] == "Seq-Bad"

    def test_error_member_failure_propagates(self, spans):
        """A member agent whose .run() itself raises propagates unchanged."""
        a, b = fake_agent("Seq-A2"), break_member_run(
            fake_agent("Seq-B2")
        )
        wf = SequentialWorkflow(
            agents=[a, b], max_loops=1, autosave=False
        )

        with pytest.raises(
            RuntimeError, match="member agent blew up"
        ):
            wf.run(task="hello")

        top = _by_name(spans, "SequentialWorkflow.run")
        assert top is not None
        assert top.status.status_code.name == "ERROR"
        a_ = _attrs(top)
        assert a_["swarms.status"] == "error"
        assert a_["swarms.error.type"] == "RuntimeError"
        assert any(e.name == "exception" for e in top.events)

        # The error propagated through the internal AgentRearrange span too.
        ar = _by_name(spans, "AgentRearrange.run")
        assert ar is not None
        assert ar.status.status_code.name == "ERROR"


# ===========================================================================
# ConcurrentWorkflow
# ===========================================================================
class TestConcurrentWorkflowTelemetry:
    def test_happy_path(self, spans):
        a, b = fake_agent("Conc-A"), fake_agent("Conc-B")
        wf = ConcurrentWorkflow(agents=[a, b], autosave=False)
        result = wf.run(task="list three colors")
        assert result

        run_spans = _all_by_name(spans, "ConcurrentWorkflow.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "ConcurrentWorkflow"
        assert a_["swarms.name"]
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "list three colors"
        assert a_["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        assert len(agent_runs) == 2

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        a, b = fake_agent("Conc-Ok"), fake_agent(
            "Conc-Bad", raise_exc=True
        )
        wf = ConcurrentWorkflow(agents=[a, b], autosave=False)
        result = wf.run(task="hello")
        assert result is not None

        top = _by_name(spans, "ConcurrentWorkflow.run")
        assert _attrs(top)["swarms.status"] == "completed"
        # No ConcurrentWorkflow.agent_error span — the future never raised,
        # because Agent.run() itself never propagated the LLM failure.
        assert (
            _by_name(spans, "ConcurrentWorkflow.agent_error") is None
        )
        assert _by_name(spans, "Agent.llm_error") is not None

    def test_error_member_failure_swallowed_by_default(self, spans):
        """Default on_error='store': swallowed per-agent, run still OK."""
        a, b = fake_agent("Conc-A2"), break_member_run(
            fake_agent("Conc-B2")
        )
        wf = ConcurrentWorkflow(agents=[a, b], autosave=False)
        result = wf.run(task="hello")
        assert result  # workflow still returns the surviving agent's output

        top = _by_name(spans, "ConcurrentWorkflow.run")
        assert top.status.status_code.name == "OK"
        assert _attrs(top)["swarms.status"] == "completed"

        err = _by_name(spans, "ConcurrentWorkflow.agent_error")
        assert err is not None
        assert err.status.status_code.name == "ERROR"
        e_ = _attrs(err)
        assert e_["swarms.error.type"] == "RuntimeError"
        assert e_["swarms.agent"] == "Conc-B2"
        assert any(ev.name == "exception" for ev in err.events)

    def test_error_member_failure_reraises_with_on_error_raise(
        self, spans
    ):
        """on_error='raise' flips the same architecture to propagate."""
        a, b = fake_agent("Conc-A3"), break_member_run(
            fake_agent("Conc-B3")
        )
        wf = ConcurrentWorkflow(
            agents=[a, b], autosave=False, on_error="raise"
        )
        with pytest.raises(
            RuntimeError, match="member agent blew up"
        ):
            wf.run(task="hello")

        top = _by_name(spans, "ConcurrentWorkflow.run")
        assert top.status.status_code.name == "ERROR"
        assert _attrs(top)["swarms.status"] == "error"
        # No separate agent_error span in this mode — it re-raises instead.
        assert (
            _by_name(spans, "ConcurrentWorkflow.agent_error") is None
        )


# ===========================================================================
# AgentRearrange
# ===========================================================================
class TestAgentRearrangeTelemetry:
    def test_happy_path_sequential_flow(self, spans):
        a, b = fake_agent("AR-A"), fake_agent("AR-B")
        ar = AgentRearrange(
            agents=[a, b],
            flow="AR-A -> AR-B",
            max_loops=1,
            autosave=False,
        )
        result = ar.run(task="draft a plan")
        assert result

        run_spans = _all_by_name(spans, "AgentRearrange.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "AgentRearrange"
        assert a_["swarms.name"]
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "draft a plan"
        assert a_["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        assert len(agent_runs) == 2

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        a, b = fake_agent("AR-Ok"), fake_agent(
            "AR-Bad", raise_exc=True
        )
        ar = AgentRearrange(
            agents=[a, b],
            flow="AR-Ok -> AR-Bad",
            max_loops=1,
            autosave=False,
        )
        result = ar.run(task="hello")
        assert result is not None

        top = _by_name(spans, "AgentRearrange.run")
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "Agent.llm_error") is not None

    def test_error_member_failure_propagates_on_sequential_flow(
        self, spans
    ):
        a, b = fake_agent("AR-A2"), break_member_run(
            fake_agent("AR-B2")
        )
        ar = AgentRearrange(
            agents=[a, b],
            flow="AR-A2 -> AR-B2",
            max_loops=1,
            autosave=False,
        )
        with pytest.raises(
            RuntimeError, match="member agent blew up"
        ):
            ar.run(task="hello")

        top = _by_name(spans, "AgentRearrange.run")
        assert top.status.status_code.name == "ERROR"
        a_ = _attrs(top)
        assert a_["swarms.status"] == "error"
        assert a_["swarms.error.type"] == "RuntimeError"
        assert any(e.name == "exception" for e in top.events)

    def test_error_member_failure_silently_swallowed_on_concurrent_flow(
        self, spans
    ):
        """Comma (parallel) flow funnels through run_agents_concurrently,
        which swallows the exception into the result value with **no**
        telemetry span at all — a real gap, not a test artifact."""
        a, b = fake_agent("AR-A3"), break_member_run(
            fake_agent("AR-B3")
        )
        ar = AgentRearrange(
            agents=[a, b],
            flow="AR-A3, AR-B3",
            max_loops=1,
            autosave=False,
        )
        result = ar.run(task="hello")  # does not raise
        assert result is not None

        top = _by_name(spans, "AgentRearrange.run")
        assert top.status.status_code.name == "OK"
        assert _attrs(top)["swarms.status"] == "completed"
        # Nothing at all marks the failure — no capture_error span exists.
        assert _by_name(spans, "AgentRearrange.error") is None
        assert _by_name(spans, "Agent.llm_error") is None


# ===========================================================================
# RoundRobinSwarm
# ===========================================================================
class TestRoundRobinSwarmTelemetry:
    def test_happy_path(self, spans):
        a, b = fake_agent("RR-A"), fake_agent("RR-B")
        rr = RoundRobinSwarm(agents=[a, b], max_loops=1)
        result = rr.run(task="pick a number")
        assert result

        run_spans = _all_by_name(spans, "RoundRobinSwarm.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "RoundRobinSwarm"
        assert a_["swarms.name"]
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "pick a number"
        assert a_["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        assert len(agent_runs) == 2

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        a, b = fake_agent("RR-Ok"), fake_agent(
            "RR-Bad", raise_exc=True
        )
        rr = RoundRobinSwarm(agents=[a, b], max_loops=1)
        result = rr.run(task="hello")
        assert result is not None

        top = _by_name(spans, "RoundRobinSwarm.run")
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "Agent.llm_error") is not None

    def test_error_member_failure_propagates(self, spans):
        a, b = fake_agent("RR-A2"), break_member_run(
            fake_agent("RR-B2")
        )
        rr = RoundRobinSwarm(agents=[a, b], max_loops=1)
        with pytest.raises(
            RuntimeError, match="member agent blew up"
        ):
            rr.run(task="hello")

        top = _by_name(spans, "RoundRobinSwarm.run")
        assert top.status.status_code.name == "ERROR"
        a_ = _attrs(top)
        assert a_["swarms.status"] == "error"
        assert a_["swarms.error.type"] == "RuntimeError"
        assert any(e.name == "exception" for e in top.events)


# ===========================================================================
# MixtureOfAgents
# ===========================================================================
class TestMixtureOfAgentsTelemetry:
    def test_happy_path(self, spans):
        workers = [fake_agent("MOA-W1"), fake_agent("MOA-W2")]
        aggregator = fake_agent("MOA-Agg", reply="synthesized answer")
        moa = MixtureOfAgents(
            agents=workers, aggregator_agent=aggregator, layers=1
        )
        result = moa.run(task="evaluate the options")
        assert result

        run_spans = _all_by_name(spans, "MixtureOfAgents.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "MixtureOfAgents"
        assert a_["swarms.name"]
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "evaluate the options"
        assert a_["swarms.output"]

        # 2 workers + 1 aggregator, all real Agents.
        agent_runs = _all_by_name(spans, "Agent.run")
        assert len(agent_runs) == 3

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        workers = [
            fake_agent("MOA-Ok"),
            fake_agent("MOA-Bad", raise_exc=True),
        ]
        aggregator = fake_agent("MOA-Agg2")
        moa = MixtureOfAgents(
            agents=workers, aggregator_agent=aggregator, layers=1
        )
        result = moa.run(task="hello")
        assert result is not None

        top = _by_name(spans, "MixtureOfAgents.run")
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "Agent.llm_error") is not None

    def test_error_member_failure_silently_swallowed(self, spans):
        """Workers run via run_agents_concurrently: a raising agent.run()
        is caught and returned as the result value with no telemetry span
        at all; the mixture still completes successfully."""
        workers = [
            fake_agent("MOA-A3"),
            break_member_run(fake_agent("MOA-B3")),
        ]
        aggregator = fake_agent("MOA-Agg3", reply="final synthesis")
        moa = MixtureOfAgents(
            agents=workers, aggregator_agent=aggregator, layers=1
        )
        result = moa.run(task="hello")  # does not raise
        assert result == "final synthesis"

        top = _by_name(spans, "MixtureOfAgents.run")
        assert top.status.status_code.name == "OK"
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "MixtureOfAgents.error") is None
        assert _by_name(spans, "Agent.llm_error") is None


# ===========================================================================
# MajorityVoting
# ===========================================================================
class TestMajorityVotingTelemetry:
    def test_happy_path(self, spans):
        voters = [fake_agent("MV-A"), fake_agent("MV-B")]
        mv = MajorityVoting(agents=voters, max_loops=1)
        # Consensus agent is auto-created by MajorityVoting; point it at a
        # fake LLM too so no real network call is ever attempted.
        mv.consensus_agent.llm = FakeLLM("consensus reached")

        result = mv.run(task="pick the best option")
        assert result

        run_spans = _all_by_name(spans, "MajorityVoting.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "MajorityVoting"
        assert a_["swarms.name"]
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "pick the best option"
        assert a_["swarms.output"]

        # 2 voters + 1 consensus agent.
        agent_runs = _all_by_name(spans, "Agent.run")
        assert len(agent_runs) == 3

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        voters = [
            fake_agent("MV-Ok"),
            fake_agent("MV-Bad", raise_exc=True),
        ]
        mv = MajorityVoting(agents=voters, max_loops=1)
        mv.consensus_agent.llm = FakeLLM("consensus reached")

        result = mv.run(task="hello")
        assert result is not None

        top = _by_name(spans, "MajorityVoting.run")
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "Agent.llm_error") is not None

    def test_error_member_failure_silently_swallowed(self, spans):
        """Voters run via run_agents_concurrently: swallowed with no span."""
        voters = [
            fake_agent("MV-A3"),
            break_member_run(fake_agent("MV-B3")),
        ]
        mv = MajorityVoting(agents=voters, max_loops=1)
        mv.consensus_agent.llm = FakeLLM("consensus reached")

        result = mv.run(task="hello")  # does not raise
        assert result is not None

        top = _by_name(spans, "MajorityVoting.run")
        assert top.status.status_code.name == "OK"
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "MajorityVoting.error") is None
        assert _by_name(spans, "Agent.llm_error") is None


# ===========================================================================
# BatchedGridWorkflow
# ===========================================================================
class TestBatchedGridWorkflowTelemetry:
    def test_happy_path(self, spans):
        a, b = fake_agent("BGW-A"), fake_agent("BGW-B")
        bgw = BatchedGridWorkflow(agents=[a, b], max_loops=1)
        tasks = ["task one", "task two"]
        result = bgw.run(tasks=tasks)
        assert result

        run_spans = _all_by_name(spans, "BatchedGridWorkflow.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "BatchedGridWorkflow"
        assert a_["swarms.name"]
        assert a_["gen_ai.operation.name"] == "swarm"
        # BatchedGridWorkflow.run(self, tasks) has no "task" parameter — the
        # captured input is "tasks", not "task".
        assert a_["swarms.input.tasks"] == str(tasks)
        assert a_["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        assert len(agent_runs) == 2

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        a, b = fake_agent("BGW-Ok"), fake_agent(
            "BGW-Bad", raise_exc=True
        )
        bgw = BatchedGridWorkflow(agents=[a, b], max_loops=1)
        result = bgw.run(tasks=["t1", "t2"])
        assert result is not None

        top = _by_name(spans, "BatchedGridWorkflow.run")
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "Agent.llm_error") is not None

    def test_error_member_failure_silently_swallowed(self, spans):
        """Runs via batched_grid_agent_execution: swallowed with no span."""
        a, b = fake_agent("BGW-A3"), break_member_run(
            fake_agent("BGW-B3")
        )
        bgw = BatchedGridWorkflow(agents=[a, b], max_loops=1)
        result = bgw.run(tasks=["t1", "t2"])  # does not raise
        assert result is not None

        top = _by_name(spans, "BatchedGridWorkflow.run")
        assert top.status.status_code.name == "OK"
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "BatchedGridWorkflow.error") is None
        assert _by_name(spans, "Agent.llm_error") is None


# ===========================================================================
# SwarmRouter — tested with two swarm_type values, since it is instrumented
# with an inline `capture_run` block rather than `@trace_run`.
# ===========================================================================
class TestSwarmRouterTelemetry:
    def test_happy_path_sequential_workflow(self, spans):
        a, b = fake_agent("SR-Seq-A"), fake_agent("SR-Seq-B")
        router = SwarmRouter(
            name="SeqRouter",
            agents=[a, b],
            swarm_type="SequentialWorkflow",
            autosave=False,
        )
        result = router.run(task="write a summary")
        assert result

        run_spans = _all_by_name(spans, "SwarmRouter.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "SwarmRouter"
        assert a_["swarms.name"] == "SeqRouter"
        assert a_["swarms.swarm_type"] == "SequentialWorkflow"
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "write a summary"
        assert a_["swarms.output"]

        assert _by_name(spans, "SequentialWorkflow.run") is not None
        assert _by_name(spans, "AgentRearrange.run") is not None
        assert len(_all_by_name(spans, "Agent.run")) == 2

    def test_happy_path_concurrent_workflow(self, spans):
        a, b = fake_agent("SR-Conc-A"), fake_agent("SR-Conc-B")
        router = SwarmRouter(
            name="ConcRouter",
            agents=[a, b],
            swarm_type="ConcurrentWorkflow",
            autosave=False,
        )
        result = router.run(task="brainstorm ideas")
        assert result

        run_spans = _all_by_name(spans, "SwarmRouter.run")
        assert len(run_spans) == 1
        span = run_spans[0]
        assert span.status.status_code.name == "OK"

        a_ = _attrs(span)
        assert a_["swarms.status"] == "completed"
        assert a_["swarms.component"] == "SwarmRouter"
        assert a_["swarms.swarm_type"] == "ConcurrentWorkflow"
        assert a_["gen_ai.operation.name"] == "swarm"
        assert a_["swarms.input.task"] == "brainstorm ideas"
        assert a_["swarms.output"]

        assert _by_name(spans, "ConcurrentWorkflow.run") is not None
        assert len(_all_by_name(spans, "Agent.run")) == 2

    def test_error_llm_raise_is_swallowed_by_agent(self, spans):
        a, b = fake_agent("SR-Ok"), fake_agent(
            "SR-Bad", raise_exc=True
        )
        router = SwarmRouter(
            name="Router3",
            agents=[a, b],
            swarm_type="SequentialWorkflow",
            autosave=False,
        )
        result = router.run(task="hello")
        assert result is not None

        top = _by_name(spans, "SwarmRouter.run")
        assert _attrs(top)["swarms.status"] == "completed"
        assert _by_name(spans, "Agent.llm_error") is not None

    def test_error_member_failure_propagates_for_sequential(
        self, spans
    ):
        a, b = fake_agent("SR-A2"), break_member_run(
            fake_agent("SR-B2")
        )
        router = SwarmRouter(
            name="Router4",
            agents=[a, b],
            swarm_type="SequentialWorkflow",
            autosave=False,
        )
        with pytest.raises(
            RuntimeError, match="member agent blew up"
        ):
            router.run(task="hello")

        top = _by_name(spans, "SwarmRouter.run")
        assert top.status.status_code.name == "ERROR"
        a_ = _attrs(top)
        assert a_["swarms.status"] == "error"
        assert a_["swarms.error.type"] == "RuntimeError"
        assert any(e.name == "exception" for e in top.events)

        # The failure propagated up through the underlying SequentialWorkflow.
        seq = _by_name(spans, "SequentialWorkflow.run")
        assert seq is not None
        assert seq.status.status_code.name == "ERROR"

    def test_error_member_failure_swallowed_for_concurrent(
        self, spans
    ):
        a, b = fake_agent("SR-A3"), break_member_run(
            fake_agent("SR-B3")
        )
        router = SwarmRouter(
            name="Router5",
            agents=[a, b],
            swarm_type="ConcurrentWorkflow",
            autosave=False,
        )
        result = router.run(task="hello")  # does not raise
        assert result is not None

        top = _by_name(spans, "SwarmRouter.run")
        assert top.status.status_code.name == "OK"
        assert _attrs(top)["swarms.status"] == "completed"

        err = _by_name(spans, "ConcurrentWorkflow.agent_error")
        assert err is not None
        assert err.status.status_code.name == "ERROR"
        assert _attrs(err)["swarms.agent"] == "SR-B3"


# ==========================================================================
# Advanced multi-agent architectures
# ==========================================================================


class TestHierarchicalSwarm:
    def _build(self):
        worker = fake_agent_final(
            "Worker1", reply="worker did the thing"
        )
        swarm = HierarchicalSwarm(
            agents=[worker],
            max_loops=1,
            director_feedback_on=False,  # avoid a second, unfaked Agent call
            agent_as_judge=False,
            planning_enabled=False,
            parallel_execution=False,
            max_agent_retries=0,
            max_reassignment_attempts=0,
            output_type="dict-all-except-first",
        )
        director_reply = json.dumps(
            {
                "plan": "Delegate the whole task to Worker1.",
                "orders": [
                    {"agent_name": "Worker1", "task": "do the thing"}
                ],
            }
        )
        swarm.director.llm = FakeLLM(reply=director_reply)
        return swarm, worker

    def test_run_emits_spans(self, spans):
        swarm, worker = self._build()
        result = swarm.run(task="Solve the puzzle")

        assert result

        span = _by_name(spans, "HierarchicalSwarm.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "HierarchicalSwarm"
        assert a["swarms.name"]
        assert a["gen_ai.operation.name"] == "swarm"
        assert a["swarms.input.task"] == "Solve the puzzle"
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        # Director + Worker1 each ran once.
        assert len(agent_runs) >= 2
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert "Worker1" in names

    def test_worker_failure_is_swallowed(self, spans):
        """Empirically verified: call_single_agent -> _execute_order_with_retries
        catches the worker's failure, records it in-band (conversation text),
        and step()/run() both swallow exceptions themselves (no re-raise) —
        so HierarchicalSwarm.run() completes normally even though a worker
        totally failed, and there is no dedicated error span for it.
        """
        swarm, worker = self._build()
        break_agent(worker, "worker exploded")

        result = swarm.run(task="Solve the puzzle")

        assert result
        span = _by_name(spans, "HierarchicalSwarm.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"

        # The failure is recorded as plain conversation text, not telemetry.
        assert "unavailable" in str(result).lower()

        # Worker1.run was bypassed entirely (its `.run` was overwritten), so
        # no Agent.run span exists for it — only the director's.
        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert "Worker1" not in names


# ===========================================================================
# MultiAgentRouter — the boss is a raw LiteLLM instance (not an Agent),
# stored as `router.function_caller`. Its `.run(task)` must return a JSON
# STRING shaped like {"handoffs": [{"reasoning", "agent_name", "task"}]}.
# ===========================================================================
class FakeBossCaller:
    def __init__(self, reply):
        self.reply = reply

    def run(self, task, *args, **kwargs):
        return self.reply


class TestMultiAgentRouter:
    def _build(self):
        analyst = fake_agent_final(
            "Analyst",
            reply="analysis complete",
            agent_description="Analyzes things",
        )
        router = MultiAgentRouter(
            agents=[analyst], output_type="dict"
        )
        return router, analyst

    def test_run_emits_spans(self, spans):
        router, analyst = self._build()
        router.function_caller = FakeBossCaller(
            json.dumps(
                {
                    "handoffs": [
                        {
                            "reasoning": "Analyst is the best fit.",
                            "agent_name": "Analyst",
                            "task": "Analyze this",
                        }
                    ]
                }
            )
        )

        result = router.run(task="Analyze this")
        assert result

        span = _by_name(spans, "MultiAgentRouter.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "MultiAgentRouter"
        assert a["gen_ai.operation.name"] == "swarm"
        assert a["swarms.input.task"] == "Analyze this"
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert "Analyst" in names

    def test_bad_boss_json_propagates(self, spans):
        """The boss's own `route_task` re-raises unwrapped (its except block
        logs then `raise`s) — a genuine propagating-error path, verified by
        reading multi_agent_router.py directly, distinct from the
        swallow-everything behavior of most member-agent failures.
        """
        router, _ = self._build()
        router.function_caller = FakeBossCaller("not valid json")

        with pytest.raises(json.JSONDecodeError):
            router.run(task="Analyze this")

        span = _by_name(spans, "MultiAgentRouter.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "ERROR"
        assert a["swarms.status"] == "error"
        assert a["swarms.error.type"] == "JSONDecodeError"
        assert any(e.name == "exception" for e in span.events)


# ===========================================================================
# GraphWorkflow — chains plain free-text agent outputs, no structured-output
# requirement at all. Nodes execute via `agent.run(prompt, img, ...)`.
# ===========================================================================
class TestGraphWorkflow:
    def _build(self):
        a = fake_agent_final("NodeA", reply="A output")
        b = fake_agent_final("NodeB", reply="B output")

        wf = GraphWorkflow(name="TestGraph", max_parallel_nodes=1)
        wf.add_node(a)
        wf.add_node(b)
        wf.add_edge(a, b)
        wf.set_entry_points([a.agent_name])
        wf.set_end_points([b.agent_name])
        return wf, a, b

    def test_run_emits_spans(self, spans):
        wf, a, b = self._build()
        result = wf.run(task="process this")

        assert result
        assert result.get("NodeB")

        span = _by_name(spans, "GraphWorkflow.run")
        assert span is not None
        attrs = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert attrs["swarms.status"] == "completed"
        assert attrs["swarms.component"] == "GraphWorkflow"
        assert attrs["gen_ai.operation.name"] == "swarm"
        assert attrs["swarms.input.task"] == "process this"
        assert attrs["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert {"NodeA", "NodeB"} <= names

    def test_node_failure_is_swallowed(self, spans):
        """GraphWorkflow's per-node collection loop catches any exception
        from a node agent and stores '[ERROR] Agent {name} failed: {e}' as
        that node's output instead of failing the run — verified directly
        in graph_workflow.py; there is no `capture_error` call anywhere in
        that file, so this failure is invisible to telemetry beyond the
        missing Agent.run span for the broken node.
        """
        wf, a, b = self._build()
        break_agent(b, "node B exploded")

        result = wf.run(task="process this")

        assert "[ERROR]" in str(result.get("NodeB", ""))

        span = _by_name(spans, "GraphWorkflow.run")
        assert span is not None
        assert span.status.status_code.name == "OK"
        assert _attrs(span)["swarms.status"] == "completed"

        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert "NodeA" in names
        assert "NodeB" not in names


# ===========================================================================
# GroupChat — every agent's LLM is forced (via RESPOND_TOOL) to answer with a
# tool-call-shaped bid: [{"function": {"name": "respond",
# "arguments": {"score": float, "message": str}}}]. Requires output_type=
# "final" (default "str-all-except-first" would stringify the list and break
# parsing) and auto_equip=False (so GroupChat doesn't rebuild `.llm`).
# ===========================================================================
def _bid(score, message):
    return [
        {
            "function": {
                "name": "respond",
                "arguments": {"score": score, "message": message},
            }
        }
    ]


class TestGroupChat:
    def _build(self):
        speaker = fake_agent_final(
            "Speaker",
            reply=_bid(0.9, "Speaker's take"),
            tools_list_dictionary=[RESPOND_TOOL],
        )
        silent = fake_agent_final(
            "Silent",
            reply=_bid(0.1, ""),
            tools_list_dictionary=[RESPOND_TOOL],
        )
        chat = GroupChat(
            agents=[speaker, silent],
            max_loops=3,
            threshold=0.5,
            auto_equip=False,
            output_type="list",
        )
        return chat, speaker, silent

    def test_run_emits_spans(self, spans):
        chat, speaker, silent = self._build()
        result = chat.run(task="Discuss autonomous swarms")

        assert result
        # Task + 2 Speaker turns (recency penalty still clears threshold).
        assert len(result) == 3

        span = _by_name(spans, "GroupChat.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "GroupChat"
        assert a["gen_ai.operation.name"] == "swarm"
        assert a["swarms.input.task"] == "Discuss autonomous swarms"
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        names = [_attrs(s).get("swarms.name") for s in agent_runs]
        assert names.count("Speaker") == 2
        assert names.count("Silent") == 2

    def test_agent_failure_is_totally_silent(self, spans):
        """`_decide_sync` catches ANY exception from `agent.run()`, logs a
        warning, and returns (0.0, "") — indistinguishable in-band from a
        legitimate "I don't want to speak" vote. `capture_error` is never
        called in groupchat.py, so a broken agent leaves *zero* telemetry
        trace of its own failure: no Agent.run span (since `.run` itself was
        replaced) and no error span anywhere. Verified empirically here.
        """
        chat, speaker, silent = self._build()
        break_agent(silent, "silent agent exploded")

        result = chat.run(task="Discuss autonomous swarms")

        assert result
        span = _by_name(spans, "GroupChat.run")
        assert span is not None
        assert span.status.status_code.name == "OK"
        assert _attrs(span)["swarms.status"] == "completed"

        agent_runs = _all_by_name(spans, "Agent.run")
        names = [_attrs(s).get("swarms.name") for s in agent_runs]
        # Speaker still ran every turn; Silent contributed zero Agent.run
        # spans because its `.run` was bypassed entirely.
        assert names.count("Speaker") == 2
        assert "Silent" not in names

        error_spans = [
            s
            for s in spans.get_finished_spans()
            if _attrs(s).get("swarms.status") == "error"
        ]
        assert error_spans == []


# ===========================================================================
# DebateWithJudge — pro/con/judge agents exchange plain free text (no
# structured output at all); the judge's raw string synthesis feeds the next
# round directly. run() has no try/except anywhere -> a member failure
# genuinely propagates (verified in debate_with_judge.py).
# ===========================================================================
class TestDebateWithJudge:
    def _build(self):
        pro = fake_agent_final("Pro-Debater", reply="Pro argument")
        con = fake_agent_final("Con-Debater", reply="Con argument")
        judge = fake_agent_final(
            "Debate-Judge", reply="Judge synthesis"
        )
        debate = DebateWithJudge(
            agents=[pro, con, judge],
            max_loops=1,
            output_type="dict-all-except-first",
            verbose=False,
        )
        return debate, pro, con, judge

    def test_run_emits_spans(self, spans):
        debate, pro, con, judge = self._build()
        result = debate.run(task="Motion: swarms beat solo agents")

        assert result
        span = _by_name(spans, "DebateWithJudge.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "DebateWithJudge"
        assert a["gen_ai.operation.name"] == "swarm"
        assert (
            a["swarms.input.task"]
            == "Motion: swarms beat solo agents"
        )
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        names = [_attrs(s).get("swarms.name") for s in agent_runs]
        # Role-intro call + one debate round call, for all three agents.
        assert names.count("Pro-Debater") == 2
        assert names.count("Con-Debater") == 2
        assert names.count("Debate-Judge") == 2

    def test_member_failure_propagates(self, spans):
        debate, pro, con, judge = self._build()
        break_agent(con, "con agent exploded")

        with pytest.raises(RuntimeError, match="con agent exploded"):
            debate.run(task="Motion: swarms beat solo agents")

        span = _by_name(spans, "DebateWithJudge.run")
        assert span is not None
        assert span.status.status_code.name == "ERROR"
        a = _attrs(span)
        assert a["swarms.status"] == "error"
        assert a["swarms.error.type"] == "RuntimeError"
        assert any(e.name == "exception" for e in span.events)


# ===========================================================================
# CouncilAsAJudge — builds its own 6 dimension-judge agents + 1 aggregator
# internally (no constructor injection point); reached post-construction via
# `council.judge_agents` (dict) / `council.aggregator_agent`. Free text only.
# Errors are caught, re-wrapped twice (DimensionEvaluationError ->
# EvaluationError), and always re-raised — never swallowed.
# ===========================================================================
class TestCouncilAsAJudge:
    def _build(self):
        council = CouncilAsAJudge(
            model_name="gpt-4o-mini",
            random_model_name=False,
            aggregation_model_name="gpt-4o-mini",
            judge_agent_model_name="gpt-4o-mini",
        )
        for dim, agent in council.judge_agents.items():
            agent.llm = FakeLLM(reply=f"{dim} rationale")
        council.aggregator_agent.llm = FakeLLM(
            reply="final aggregated report"
        )
        return council

    def test_run_emits_spans(self, spans):
        council = self._build()
        result = council.run(
            task="Evaluate this response for quality."
        )

        assert result
        span = _by_name(spans, "CouncilAsAJudge.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "CouncilAsAJudge"
        assert a["gen_ai.operation.name"] == "swarm"
        assert (
            a["swarms.input.task"]
            == "Evaluate this response for quality."
        )
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        # 6 dimension judges + 1 aggregator.
        assert len(agent_runs) == len(council.judge_agents) + 1

    def test_dimension_failure_propagates_as_evaluation_error(
        self, spans
    ):
        council = self._build()
        first_dim = next(iter(council.judge_agents))
        break_agent(council.judge_agents[first_dim], "judge exploded")

        with pytest.raises(EvaluationError):
            council.run(task="Evaluate this response for quality.")

        span = _by_name(spans, "CouncilAsAJudge.run")
        assert span is not None
        assert span.status.status_code.name == "ERROR"
        a = _attrs(span)
        assert a["swarms.status"] == "error"
        assert a["swarms.error.type"] == "EvaluationError"
        assert any(e.name == "exception" for e in span.events)


# ===========================================================================
# LLMCouncil — accepts pre-built council members via `council_members=[...]`;
# the Chairman is always built internally, reached via `council.chairman`.
# Free text throughout; a council member's failure is swallowed (the
# exception object itself is substituted as its "response"), but a Chairman
# failure is NOT wrapped in any try/except -> genuinely propagates.
# ===========================================================================
class TestLLMCouncil:
    def _build(self):
        m1 = fake_agent_final("Member-1", reply="Member 1 answer")
        m2 = fake_agent_final("Member-2", reply="Member 2 answer")
        council = LLMCouncil(
            council_members=[m1, m2],
            chairman_model="gpt-4o-mini",
            verbose=False,
            output_type="dict-all-except-first",
        )
        council.chairman.llm = FakeLLM(reply="chairman synthesis")
        return council, m1, m2

    def test_run_emits_spans(self, spans):
        council, m1, m2 = self._build()
        result = council.run(task="Discuss microservice architecture")

        assert result
        span = _by_name(spans, "LLMCouncil.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "LLMCouncil"
        assert a["gen_ai.operation.name"] == "swarm"
        assert (
            a["swarms.input.task"]
            == "Discuss microservice architecture"
        )
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        names = [_attrs(s).get("swarms.name") for s in agent_runs]
        # Each member answers once, then evaluates once; chairman once.
        assert names.count("Member-1") == 2
        assert names.count("Member-2") == 2
        assert names.count("Chairman") == 1

    def test_chairman_failure_propagates(self, spans):
        council, m1, m2 = self._build()
        break_agent(council.chairman, "chairman exploded")

        with pytest.raises(RuntimeError, match="chairman exploded"):
            council.run(task="Discuss microservice architecture")

        span = _by_name(spans, "LLMCouncil.run")
        assert span is not None
        assert span.status.status_code.name == "ERROR"
        a = _attrs(span)
        assert a["swarms.status"] == "error"
        assert a["swarms.error.type"] == "RuntimeError"
        assert any(e.name == "exception" for e in span.events)


# ===========================================================================
# HeavySwarm — always builds its own agents internally as `self.agents`
# (Dict[str, Agent] keyed by role for the "default" variant: research,
# analysis, alternatives, verification, synthesis). Question generation uses
# a throwaway, non-instance LiteLLM object, so it can't be faked via `.llm`;
# `execute_question_generation` is monkeypatched directly instead (its only
# contract is returning a dict of `*_question` keys). Worker agents are
# hard-coded to `max_loops="auto"` internally and must be forced to an int
# after construction to avoid spinning forever against a scripted FakeLLM.
# Every stage of HeavySwarm.run() catches its own exceptions and returns an
# error STRING instead of raising -> it essentially never raises, and never
# calls capture_error.
# ===========================================================================
class TestHeavySwarm:
    def _build(self):
        swarm = HeavySwarm(
            variant="default",
            worker_model_name="gpt-4o-mini",
            question_agent_model_name="gpt-4o-mini",
            max_loops=1,
            show_dashboard=False,
            verbose=False,
            agent_prints_on=False,
            output_type="dict-all-except-first",
            timeout=15,
        )
        for key, agent in swarm.agents.items():
            agent.llm = FakeLLM(reply=f"{key} agent output")
            agent.max_loops = (
                1  # override hard-coded "auto" for workers
            )

        def fake_question_generation(task):
            return {
                "thinking": "straightforward task",
                "research_question": "What are the facts?",
                "analysis_question": "What does it mean?",
                "alternatives_question": "What else could we do?",
                "verification_question": "Is this correct?",
            }

        swarm.execute_question_generation = fake_question_generation
        return swarm

    def test_run_emits_spans(self, spans):
        swarm = self._build()
        result = swarm.run(task="Analyze the coffee market")

        assert result
        span = _by_name(spans, "HeavySwarm.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "HeavySwarm"
        assert a["gen_ai.operation.name"] == "swarm"
        assert a["swarms.input.task"] == "Analyze the coffee market"
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        expected_names = {
            agent.agent_name for agent in swarm.agents.values()
        }
        assert expected_names <= names

    def test_worker_failure_is_dropped_silently(self, spans):
        """Every stage of HeavySwarm.run() wraps its own try/except; a
        worker's exception is actually caught *inside* `execute_agent`
        itself (heavy_swarm.py), which returns `(agent_type, "Error: ...")`
        instead of raising. That tuple lands in the internal `results` dict,
        but `_synthesize_results` never reads that dict's values — it only
        reads `self.conversation`, and `conversation.add(...)` for that
        agent is skipped on the exception path. Net effect, verified
        empirically here: the failed worker's error text does not even
        reach the final synthesized output, let alone telemetry — it is
        only visible via `self._log(...)` (stdout), and `capture_error` is
        never called anywhere in heavy_swarm.py.
        """
        swarm = self._build()
        research_agent = swarm.agents["research"]
        break_agent(research_agent, "research agent exploded")

        result = swarm.run(task="Analyze the coffee market")

        assert result
        # The broken agent's name/output never made it into the final
        # conversation-derived output at all — not even as an error string.
        assert research_agent.agent_name not in str(result)

        span = _by_name(spans, "HeavySwarm.run")
        assert span is not None
        assert span.status.status_code.name == "OK"
        assert _attrs(span)["swarms.status"] == "completed"

        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert research_agent.agent_name not in names
        assert swarm.agents["synthesis"].agent_name in names


# ===========================================================================
# PlannerWorkerSwarm — worker Agents are accepted pre-built via
# `agents=[...]`, but the planner and judge agents are always freshly built
# internally each cycle (`_create_planner_agent` / `_run_judge`), with no
# constructor injection point. Faked by monkeypatching those two instance
# methods directly. The planner's structured output is a JSON string/dict
# parsed by `_parse_structured_output` into a `PlannerTaskSpec`
# ({"plan": str, "tasks": [{"title", "description", "priority",
# "depends_on_titles"}]}) — plain dict/JSON-string is accepted directly, no
# tool-call wire format required.
# ===========================================================================
class TestPlannerWorkerSwarm:
    def _build(self):
        worker = fake_agent_final(
            "Worker1", reply="task completed successfully"
        )
        swarm = PlannerWorkerSwarm(
            agents=[worker],
            planner_model_name="gpt-4o-mini",
            judge_model_name="gpt-4o-mini",
            max_loops=1,
            output_type="dict-all-except-first",
            worker_timeout=15,
            task_timeout=10,
        )

        plan_reply = json.dumps(
            {
                "plan": "One task, assign it to the only worker.",
                "tasks": [
                    {
                        "title": "T1",
                        "description": "Do the thing",
                        "priority": 1,
                        "depends_on_titles": [],
                    }
                ],
            }
        )

        def fake_create_planner_agent(name="Planner"):
            return fake_agent_final(name, reply=plan_reply)

        # `_run_judge` is called as `self._run_judge(img=img)`; the double
        # must accept it or every PlannerWorkerSwarm test raises TypeError.
        def fake_run_judge(img=None, **kwargs):
            verdict = CycleVerdict(
                is_complete=True,
                overall_quality=9,
                summary="Goal achieved.",
                gaps=[],
                follow_up_instructions=None,
                needs_fresh_start=False,
            )
            swarm.conversation.add(
                role="CycleJudge",
                content=(
                    f"Quality: {verdict.overall_quality}/10 | "
                    f"Complete: {verdict.is_complete}\n{verdict.summary}"
                ),
            )
            return verdict

        swarm._create_planner_agent = fake_create_planner_agent
        swarm._run_judge = fake_run_judge
        return swarm, worker

    def test_run_emits_spans(self, spans):
        swarm, worker = self._build()
        result = swarm.run(task="Build a widget")

        assert result
        span = _by_name(spans, "PlannerWorkerSwarm.run")
        assert span is not None
        a = _attrs(span)
        assert span.status.status_code.name == "OK"
        assert a["swarms.status"] == "completed"
        assert a["swarms.component"] == "PlannerWorkerSwarm"
        assert a["gen_ai.operation.name"] == "swarm"
        assert a["swarms.input.task"] == "Build a widget"
        assert a["swarms.output"]

        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert "Planner" in names
        assert "Worker1" in names

    def test_worker_failure_is_swallowed(self, spans):
        """WorkerPool._worker_loop catches any exception from a worker's
        `.run()`, logs it, and marks the task failed/retried via the task
        queue rather than raising -> PlannerWorkerSwarm.run() still
        completes normally. `capture_error` is never called in
        planner_worker_swarm.py.
        """
        swarm, worker = self._build()
        break_agent(worker, "worker exploded")

        result = swarm.run(task="Build a widget")

        assert result
        span = _by_name(spans, "PlannerWorkerSwarm.run")
        assert span is not None
        assert span.status.status_code.name == "OK"
        assert _attrs(span)["swarms.status"] == "completed"

        agent_runs = _all_by_name(spans, "Agent.run")
        names = {_attrs(s).get("swarms.name") for s in agent_runs}
        assert "Planner" in names
        assert "Worker1" not in names


# ==========================================================================
# Identity helpers
# ==========================================================================


class TestUserIdentity:
    """generate_user_id and get_machine_id."""

    def test_generate_user_id_returns_a_uuid4_string(self):
        user_id = generate_user_id()
        assert isinstance(user_id, str)
        assert uuid.UUID(user_id, version=4)

    def test_generated_user_ids_are_unique(self):
        assert len({generate_user_id() for _ in range(100)}) == 100

    def test_machine_id_is_a_sha256_hex_digest(self):
        machine_id = get_machine_id()
        assert isinstance(machine_id, str)
        assert len(machine_id) == 64
        assert all(c in "0123456789abcdef" for c in machine_id)

    def test_machine_id_is_stable_across_calls(self):
        assert len({get_machine_id() for _ in range(100)}) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
