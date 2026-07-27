"""Offline OTel telemetry tests for swarms' more complex multi-agent architectures.

Covers: HierarchicalSwarm, MultiAgentRouter, GraphWorkflow, GroupChat,
DebateWithJudge, CouncilAsAJudge, LLMCouncil, HeavySwarm, PlannerWorkerSwarm.

Every LLM call in this file is faked (no network, no API key). Each architecture
drives at least one director/router/judge/planner agent that must return
STRUCTURED output for the orchestration to proceed; a ``FakeLLM`` is scripted
per-test to return exactly the shape that architecture's own parsing code
expects (traced by reading each ``swarms/structs/*.py`` source directly — see
inline comments for the exact contract).

Run with:
    PYTHONPATH=. python3 -m pytest tests/telemetry/test_telemetry_multi_agent_advanced.py -q -p no:randomly
"""

import json
import os

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import swarms.telemetry.otel as otel
from swarms import (
    Agent,
    CouncilAsAJudge,
    GraphWorkflow,
    GroupChat,
    HeavySwarm,
    HierarchicalSwarm,
    MultiAgentRouter,
    RESPOND_TOOL,
)
from swarms.schemas.planner_worker_schemas import CycleVerdict
from swarms.structs.council_as_judge import EvaluationError
from swarms.structs.debate_with_judge import DebateWithJudge
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm

# ---------------------------------------------------------------------------
# Fixtures — same pattern as tests/telemetry/test_telemetry.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _exporter():
    os.environ["SWARMS_TELEMETRY_ON"] = "true"
    otel.TELEMETRY_BASE_URL = "http://127.0.0.1:9/dead-multi-agent-advanced"  # unreachable: nothing leaves the machine
    otel.swarm_telemetry.cache_clear()
    telem = otel.swarm_telemetry()
    assert telem.ready
    mem = InMemorySpanExporter()
    telem._provider.add_span_processor(SimpleSpanProcessor(mem))
    return mem


@pytest.fixture
def spans(_exporter):
    _exporter.clear()
    return _exporter


def _by_name(exporter, name):
    return next(
        (s for s in exporter.get_finished_spans() if s.name == name),
        None,
    )


def _all_by_name(exporter, name):
    return [
        s for s in exporter.get_finished_spans() if s.name == name
    ]


def _attrs(span):
    return dict(span.attributes)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Stand-in for LiteLLM: no network, deterministic reply.

    ``reply`` can be any object (str, dict, list) — ``Agent`` only forwards it
    verbatim (see ``LLMManager.call``'s non-streaming path), so callers pick
    whatever shape the consuming orchestrator's parser expects.
    """

    def __init__(self, reply="FAKE OUTPUT"):
        self.stream = False
        self.temperature = 0.5
        self.reply = reply

    def run(self, task=None, img=None, **kwargs):
        return self.reply


def fake_agent(name, reply=None, **kwargs):
    """Build a real ``Agent`` wired to a ``FakeLLM``.

    ``output_type="final"`` is used everywhere in this file (rather than the
    framework default ``"str-all-except-first"``) so ``agent.run()`` always
    returns the raw ``FakeLLM.reply`` object unchanged (via
    ``Conversation.get_final_message_content``) — critical for the
    architectures that need a structured (list/dict) reply to survive
    untouched, and harmless free-text passthrough for the ones that don't.
    """
    settings = dict(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        verbose=False,
        output_type="final",
    )
    settings.update(kwargs)
    agent = Agent(**settings)
    agent.llm = FakeLLM(
        reply if reply is not None else f"{name} output"
    )
    return agent


def break_agent(agent, message="synthetic agent failure"):
    """Make ``agent.run(...)`` raise directly, bypassing ``Agent``'s own
    LLM-error swallowing.

    ``Agent._run`` catches everything ``self.llm.run()`` raises, retries, and
    then returns normally without re-raising (see
    ``TestRealLLMRuns.test_agent_run_error_is_logged`` in
    ``tests/telemetry/test_telemetry.py``) — so a raising ``FakeLLM`` alone
    never lets an exception reach the swarm's own orchestration code. To
    empirically observe how each *swarm* (not ``Agent``) handles a member
    that is completely broken, the bound ``.run`` method is replaced outright.
    This also means no ``Agent.run`` span is emitted for this agent at all
    (the ``@trace_run`` decorator lives on the class method we just shadowed).
    """

    def _raise(*args, **kwargs):
        raise RuntimeError(message)

    agent.run = _raise
    return agent


# ===========================================================================
# HierarchicalSwarm — director must return a SwarmSpec-shaped
# {"plan": str, "orders": [{"agent_name": str, "task": str}, ...]}, consumed
# by HierarchicalSwarm.parse_orders (accepts a JSON string, dict, or list).
# ===========================================================================
class TestHierarchicalSwarm:
    def _build(self):
        worker = fake_agent("Worker1", reply="worker did the thing")
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
        analyst = fake_agent(
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
        a = fake_agent("NodeA", reply="A output")
        b = fake_agent("NodeB", reply="B output")

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
        speaker = fake_agent(
            "Speaker",
            reply=_bid(0.9, "Speaker's take"),
            tools_list_dictionary=[RESPOND_TOOL],
        )
        silent = fake_agent(
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
        pro = fake_agent("Pro-Debater", reply="Pro argument")
        con = fake_agent("Con-Debater", reply="Con argument")
        judge = fake_agent("Debate-Judge", reply="Judge synthesis")
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
        m1 = fake_agent("Member-1", reply="Member 1 answer")
        m2 = fake_agent("Member-2", reply="Member 2 answer")
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
        worker = fake_agent(
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
            return fake_agent(name, reply=plan_reply)

        def fake_run_judge():
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


if __name__ == "__main__":
    pytest.main()
