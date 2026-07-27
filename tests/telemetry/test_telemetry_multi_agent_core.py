"""Offline, end-to-end OpenTelemetry coverage for the core multi-agent
architectures.

Companion to ``test_telemetry.py`` (which covers the primitives, static
instrumentation coverage across every architecture, and init spans for a
handful of classes). This file instead *runs* eight core architectures
end-to-end — ``SequentialWorkflow``, ``ConcurrentWorkflow``,
``AgentRearrange``, ``RoundRobinSwarm``, ``MixtureOfAgents``,
``MajorityVoting``, ``BatchedGridWorkflow``, and ``SwarmRouter`` (two
``swarm_type`` values) — against a fake, network-free LLM and asserts the
resulting span tree.

Everything here is 100% offline: ``Agent.llm`` is replaced with an in-memory
stand-in before any ``run()`` call, so no LiteLLM/network call is ever made
and no API key is required.

Error-path methodology (read before editing)
---------------------------------------------
Two distinct fault-injection techniques are used, because they exercise
different layers and are **not interchangeable**:

1. ``FakeLLM.run`` raising. This is the injection point suggested by the
   task brief, so every section includes one such test. Empirically (see
   ``TestAgentSwallowsLLMErrors`` and the mirrored per-architecture tests)
   this is *always* swallowed by ``Agent.run``'s own retry loop
   (``swarms/structs/agent.py``, the ``while attempt < self.retry_attempts``
   block catches bare ``Exception``) before it ever reaches the calling
   architecture. The observable effect is uniform across every architecture
   tested here: the member ``Agent.run`` span still reports
   ``swarms.status == "completed"``, a separate ``Agent.llm_error`` span is
   emitted via ``capture_error``, and the parent ``<Class>.run`` span is
   unaffected (still ``completed`` / OK). This is a real and slightly
   surprising finding: no multi-agent structure's own error-handling code
   (``ConcurrentWorkflow``'s ``on_error`` branch, ``AgentRearrange``'s
   ``_catch_error``, etc.) is ever reached via this route.

2. Patching the member ``Agent`` instance's bound ``.run`` attribute directly
   (``agent.run = <raising callable>``), bypassing ``Agent.run`` entirely.
   This simulates a failure the swarm's own orchestration code has to deal
   with directly (not one absorbed upstream by ``Agent`` itself), and is the
   only way to actually exercise — and empirically tell apart — each
   architecture's own error handling:
        - ``SequentialWorkflow`` / ``AgentRearrange`` (sequential flow) /
          ``RoundRobinSwarm`` / ``SwarmRouter(swarm_type="SequentialWorkflow")``:
          the exception propagates all the way out (re-raised), the run span
          gets ``ERROR`` status with ``swarms.error.type`` set and an
          ``exception`` event, and every span it passes through on the way up
          also flips to ``ERROR``.
        - ``ConcurrentWorkflow`` (default ``on_error="store"``) /
          ``SwarmRouter(swarm_type="ConcurrentWorkflow")``: the failure is
          swallowed per-agent and re-emitted as a standalone
          ``ConcurrentWorkflow.agent_error`` span (via ``capture_error``); the
          parent run span still completes successfully.
        - ``ConcurrentWorkflow(on_error="raise")``: same architecture, opposite
          behavior — re-raises instead of swallowing.
        - ``AgentRearrange`` (concurrent/comma flow) / ``MixtureOfAgents`` /
          ``MajorityVoting`` / ``BatchedGridWorkflow``: all funnel through
          ``run_agents_concurrently`` / ``batched_grid_agent_execution``
          (``swarms/structs/multi_agent_exec.py``), which catches the
          exception and returns it *as the result value* with **no**
          telemetry span at all — not even a ``capture_error``. The failure is
          completely invisible to telemetry; it only shows up as a stringified
          exception object embedded in the swarm's output.

Both techniques are tested for every architecture below so the suite
documents the real, empirically-verified behavior rather than assuming a
uniform contract.
"""

import os

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import swarms.telemetry.otel as otel
from swarms import (
    Agent,
    AgentRearrange,
    ConcurrentWorkflow,
    MajorityVoting,
    MixtureOfAgents,
    RoundRobinSwarm,
    SequentialWorkflow,
    SwarmRouter,
)
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow

# ---------------------------------------------------------------------------
# Fixtures — same pattern as tests/telemetry/test_telemetry.py
# ---------------------------------------------------------------------------
os.environ["SWARMS_OTEL_TIMEOUT"] = "2"


@pytest.fixture(scope="session")
def _exporter():
    """Build the telemetry singleton and attach an in-memory span exporter.

    Points the module-level base URL at a dead local address *before* the
    provider is constructed, so no span emitted by this file ever leaves the
    machine, and rebuilds the ``lru_cache``d singleton so the env gate is
    re-read with ``SWARMS_TELEMETRY_ON`` set (in-process, not via the shell
    environment — the harness never relies on an externally-set env var).
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
    """Clear captured spans before each test; yield the exporter for reading."""
    _exporter.clear()
    return _exporter


def _by_name(exporter, name):
    """Return the single finished span named ``name``, or ``None``."""
    return next(
        (s for s in exporter.get_finished_spans() if s.name == name),
        None,
    )


def _all_by_name(exporter, name):
    """Return every finished span named ``name``, in export order."""
    return [
        s for s in exporter.get_finished_spans() if s.name == name
    ]


def _attrs(span):
    return dict(span.attributes)


# ---------------------------------------------------------------------------
# Offline fake LLM — no network, deterministic, optionally raises.
# ---------------------------------------------------------------------------
class FakeLLM:
    """Stand-in for LiteLLM: no network, deterministic reply (or raises)."""

    def __init__(self, reply="FAKE OUTPUT", raise_exc=False):
        self.stream = False
        self.temperature = 0.5
        self.reply = reply
        self.raise_exc = raise_exc

    def run(self, task=None, img=None, **kwargs):
        if self.raise_exc:
            raise RuntimeError(f"llm blew up for task: {task!r}"[:80])
        return self.reply


def fake_agent(name, reply=None, raise_exc=False):
    """Build a real ``Agent`` wired to a network-free ``FakeLLM``.

    ``retry_attempts=1`` keeps the (still-offline) internal retry loop from
    looping 3x on the error-path tests.
    """
    a = Agent(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        verbose=False,
        retry_attempts=1,
    )
    a.llm = FakeLLM(reply or f"{name} output", raise_exc=raise_exc)
    return a


def break_member_run(agent, message="member agent blew up"):
    """Replace ``agent.run`` itself with a raiser, bypassing Agent.run().

    Unlike a raising ``FakeLLM``, this is *not* absorbed by ``Agent.run``'s
    own retry loop — it simulates a failure the calling swarm structure must
    handle directly, which is the only way to exercise (and tell apart) each
    architecture's own error-handling code. See the module docstring.
    """

    def _raiser(*args, **kwargs):
        raise RuntimeError(message)

    agent.run = _raiser
    return agent


# ===========================================================================
# SequentialWorkflow
# ===========================================================================
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


if __name__ == "__main__":
    pytest.main()
