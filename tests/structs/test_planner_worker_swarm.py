import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from swarms import Agent
from swarms.prompts.planner_worker_prompts import WORKER_SYSTEM_PROMPT
from swarms.schemas.planner_worker_schemas import (
    CycleVerdict,
    PlannerTask,
    PlannerTaskSpec,
    PlannerTaskOutput,
    PlannerTaskStatus,
    TaskPriority,
)
from swarms.structs.planner_worker_swarm import (
    PlannerWorkerSwarm,
    TaskQueue,
    WorkerPool,
)
from swarms.structs.conversation import Conversation


# -- helpers --

def _agent(name="W"):
    return Agent(
        agent_name=name, agent_description=name,
        model_name="gpt-4o-mini", max_loops=1,
        verbose=False, print_on=False,
    )

def _mock_agent(name="W"):
    a = MagicMock(spec=Agent)
    a.agent_name = name
    a.run = MagicMock(return_value=f"result-from-{name}")
    a.short_memory_init = MagicMock(return_value="fresh")
    return a

def _swarm(max_loops=1, **kw):
    return PlannerWorkerSwarm(agents=[_agent("W0"), _agent("W1")], max_loops=max_loops, **kw)

def _drain(q, task_id):
    """Claim→start→complete a single task through the queue."""
    c = q.claim("w")
    q.start(c.id, c.version)
    cur = q.get_task(c.id)
    q.complete(c.id, "done", cur.version)


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------
class TestTaskQueue:
    def test_add_and_len(self):
        q = TaskQueue()
        t = PlannerTask(title="T", description="D")
        assert q.add_task(t) == t.id
        assert len(q) == 1

    def test_bulk_add(self):
        q = TaskQueue()
        ids = q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(5)])
        assert len(ids) == 5

    def test_claim_highest_priority(self):
        q = TaskQueue()
        q.add_tasks([
            PlannerTask(title="Low", description="d", priority=TaskPriority.LOW),
            PlannerTask(title="High", description="d", priority=TaskPriority.HIGH),
            PlannerTask(title="Normal", description="d", priority=TaskPriority.NORMAL),
        ])
        assert q.claim("w").title == "High"

    def test_claim_respects_deps(self):
        q = TaskQueue()
        a = PlannerTask(title="A", description="first")
        b = PlannerTask(title="B", description="second", depends_on=[a.id])
        q.add_tasks([a, b])
        claimed = q.claim("w")
        assert claimed.title == "A"  # B blocked
        q.start(claimed.id, claimed.version)
        cur = q.get_task(claimed.id)
        q.complete(claimed.id, "done", cur.version)
        assert q.claim("w").title == "B"

    def test_claim_empty(self):
        assert TaskQueue().claim("w") is None

    def test_no_double_claims_50_threads(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(20)])
        claimed = []
        lock = threading.Lock()
        def claimer(name):
            while (t := q.claim(name)):
                q.start(t.id, t.version)
                with lock: claimed.append(t.id)
        threads = [threading.Thread(target=claimer, args=(f"w{i}",)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(claimed) == len(set(claimed)) == 20

    def test_version_check(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="T", description="D"))
        c = q.claim("w")
        assert q.start(c.id, c.version + 999) is False
        assert q.start(c.id, c.version) is True

    def test_complete(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="T", description="D"))
        c = q.claim("w"); q.start(c.id, c.version)
        cur = q.get_task(c.id)
        assert q.complete(c.id, "result!", cur.version)
        f = q.get_task(c.id)
        assert f.status == PlannerTaskStatus.COMPLETED and f.result == "result!"

    def test_fail_retry(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="T", description="D", max_retries=2))
        c = q.claim("w"); q.start(c.id, c.version)
        cur = q.get_task(c.id)
        q.fail(c.id, "oops", cur.version)
        assert q.get_task(c.id).status == PlannerTaskStatus.PENDING

    def test_fail_permanent(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="T", description="D", max_retries=0))
        c = q.claim("w"); q.start(c.id, c.version)
        cur = q.get_task(c.id)
        q.fail(c.id, "fatal", cur.version)
        assert q.get_task(c.id).status == PlannerTaskStatus.FAILED

    def test_is_all_done(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title="T1", description="D"), PlannerTask(title="T2", description="D")])
        assert not q.is_all_done()
        for _ in range(2): _drain(q, None)
        assert q.is_all_done()

    def test_cancel(self):
        q = TaskQueue()
        t = PlannerTask(title="T", description="D")
        q.add_task(t)
        assert q.cancel(t.id)
        assert q.get_task(t.id).status == PlannerTaskStatus.CANCELLED

    def test_dependency_chain_abc(self):
        q = TaskQueue()
        a = PlannerTask(title="A", description="1")
        b = PlannerTask(title="B", description="2", depends_on=[a.id])
        c = PlannerTask(title="C", description="3", depends_on=[b.id])
        q.add_tasks([a, b, c])
        for expected in ["A", "B", "C"]:
            cl = q.claim("w")
            assert cl.title == expected
            q.start(cl.id, cl.version)
            cur = q.get_task(cl.id)
            q.complete(cl.id, f"{expected}-done", cur.version)

    def test_clear(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(5)])
        assert q.clear() == 5
        assert len(q) == 0

    def test_clear_non_terminal_preserves_completed(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title="A", description="D"), PlannerTask(title="B", description="D")])
        _drain(q, None)  # completes A
        assert q.clear_non_terminal() == 1
        assert len(q) == 1
        assert q.get_all_tasks()[0].status == PlannerTaskStatus.COMPLETED

    def test_claim_returns_copy(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="T", description="D"))
        c = q.claim("w")
        c.title = "MODIFIED"
        assert q.get_task(c.id).title == "T"

    def test_dependency_results(self):
        q = TaskQueue()
        a = PlannerTask(title="Setup", description="d")
        b = PlannerTask(title="Build", description="d", depends_on=[a.id])
        q.add_tasks([a, b])
        c = q.claim("w"); q.start(c.id, c.version)
        cur = q.get_task(c.id); q.complete(c.id, "setup-ok", cur.version)
        assert "setup-ok" in q.get_dependency_results(b.id)

    def test_status_report(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title="A", description="d"), PlannerTask(title="B", description="d")])
        _drain(q, None)
        s = q.get_status()
        assert s["total"] == 2 and s["progress"] == "1/2"

    def test_200_tasks_100_threads(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(200)])
        claimed = []; lock = threading.Lock()
        def claimer(name):
            while (t := q.claim(name)):
                q.start(t.id, t.version)
                cur = q.get_task(t.id); q.complete(t.id, "r", cur.version)
                with lock: claimed.append(t.id)
        threads = [threading.Thread(target=claimer, args=(f"w{i}",)) for i in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(claimed) == len(set(claimed)) == 200

    def test_diamond_dependency(self):
        q = TaskQueue()
        a = PlannerTask(title="A", description="root")
        b = PlannerTask(title="B", description="l", depends_on=[a.id])
        c = PlannerTask(title="C", description="r", depends_on=[a.id])
        d = PlannerTask(title="D", description="join", depends_on=[b.id, c.id])
        q.add_tasks([a, b, c, d])
        _drain(q, None)  # A
        names = {q.claim("w").title for _ in range(2)}  # B, C become available
        assert names == {"B", "C"}

    def test_retry_exhaustion(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="Flaky", description="d", max_retries=2))
        for _ in range(3):
            c = q.claim("w"); q.start(c.id, c.version)
            cur = q.get_task(c.id); q.fail(c.id, "err", cur.version)
        assert q.get_task(q.get_all_tasks()[0].id).status == PlannerTaskStatus.FAILED


# ---------------------------------------------------------------------------
# Structured output parsing
# ---------------------------------------------------------------------------
class TestParsing:
    def test_dict(self):
        r = _swarm()._parse_structured_output(
            {"plan": "P", "tasks": [{"title": "T", "description": "D", "priority": 1, "depends_on_titles": []}]},
            PlannerTaskSpec)
        assert isinstance(r, PlannerTaskSpec)

    def test_json_string(self):
        import json
        r = _swarm()._parse_structured_output(
            json.dumps({"plan": "P", "tasks": [{"title": "T", "description": "D", "priority": 1, "depends_on_titles": []}]}),
            PlannerTaskSpec)
        assert isinstance(r, PlannerTaskSpec)

    def test_function_call_format(self):
        import json
        r = _swarm()._parse_structured_output(
            [{"function": {"name": "PlannerTaskSpec", "arguments": json.dumps(
                {"plan": "P", "tasks": [{"title": "T", "description": "D", "priority": 0, "depends_on_titles": []}]})}}],
            PlannerTaskSpec)
        assert r.plan == "P"

    def test_conversation_format(self):
        import json
        r = _swarm()._parse_structured_output(
            [{"role": "assistant", "content": [{"function": {"name": "X", "arguments": json.dumps(
                {"plan": "N", "tasks": [{"title": "T", "description": "D", "priority": 1, "depends_on_titles": []}]})}}]}],
            PlannerTaskSpec)
        assert r.plan == "N"

    def test_model_instance_passthrough(self):
        spec = PlannerTaskSpec(plan="P", tasks=[PlannerTaskOutput(title="T", description="D", priority=1)])
        assert _swarm()._parse_structured_output(spec, PlannerTaskSpec) is spec

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _swarm()._parse_structured_output(12345, PlannerTaskSpec)

    def test_cycle_verdict(self):
        r = _swarm()._parse_structured_output(
            {"is_complete": True, "overall_quality": 8, "summary": "OK", "needs_fresh_start": False},
            CycleVerdict)
        assert r.is_complete and not r.needs_fresh_start


# ---------------------------------------------------------------------------
# WorkerPool (mocked — no API)
# ---------------------------------------------------------------------------
class TestWorkerPoolMocked:
    def test_drains_queue(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description=f"D{i}") for i in range(3)])
        a = _mock_agent("W1")
        WorkerPool(agents=[a], task_queue=q, conversation=Conversation(time_enabled=False), max_workers=1).run(timeout=10)
        assert q.is_all_done() and a.run.call_count == 3

    def test_worker_prompt_injected(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="MyTask", description="Do it"))
        a = _mock_agent("W1")
        WorkerPool(agents=[a], task_queue=q, conversation=Conversation(time_enabled=False), max_workers=1).run(timeout=10)
        task_text = a.run.call_args[1]["task"]
        assert "Worker Agent" in task_text and "MyTask" in task_text

    def test_memory_reset_per_task(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(3)])
        a = _mock_agent("W1")
        WorkerPool(agents=[a], task_queue=q, conversation=Conversation(time_enabled=False), max_workers=1).run(timeout=10)
        assert a.short_memory_init.call_count == 3

    def test_independent_workers(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(6)])
        agents = [_mock_agent(f"W{i}") for i in range(3)]
        WorkerPool(agents=agents, task_queue=q, conversation=Conversation(time_enabled=False), max_workers=3).run(timeout=10)
        assert q.is_all_done() and sum(a.run.call_count for a in agents) == 6

    def test_dep_context_passed(self):
        q = TaskQueue()
        a = PlannerTask(title="Research", description="Do research")
        b = PlannerTask(title="Summarize", description="Sum", depends_on=[a.id])
        q.add_tasks([a, b])
        agent = _mock_agent("W1")
        agent.run = MagicMock(side_effect=["research-data", "summary"])
        WorkerPool(agents=[agent], task_queue=q, conversation=Conversation(time_enabled=False), max_workers=1).run(timeout=10)
        assert "research-data" in agent.run.call_args_list[1][1]["task"]

    def test_retry_on_failure(self):
        q = TaskQueue()
        q.add_task(PlannerTask(title="Flaky", description="d", max_retries=1))
        n = [0]
        def fx(**kw):
            n[0] += 1
            if n[0] == 1: raise RuntimeError("fail")
            return "ok"
        a = _mock_agent("W1"); a.run = MagicMock(side_effect=fx)
        WorkerPool(agents=[a], task_queue=q, conversation=Conversation(time_enabled=False), max_workers=1).run(timeout=10)
        assert q.is_all_done() and list(q.get_results_summary().values()) == ["ok"]

    def test_pool_timeout(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(100)])
        a = _mock_agent("W1"); a.run = MagicMock(side_effect=lambda **kw: time.sleep(0.5) or "done")
        t0 = time.time()
        WorkerPool(agents=[a], task_queue=q, conversation=Conversation(time_enabled=False), max_workers=1).run(timeout=1.5)
        assert time.time() - t0 < 5 and not q.is_all_done()

    def test_stop_signal(self):
        q = TaskQueue()
        q.add_tasks([PlannerTask(title=f"T{i}", description="D") for i in range(50)])
        a = _mock_agent("W1"); a.run = MagicMock(side_effect=lambda **kw: time.sleep(0.2) or "done")
        pool = WorkerPool(agents=[a], task_queue=q, conversation=Conversation(time_enabled=False), max_workers=1)
        threading.Timer(0.5, pool.stop).start()
        t0 = time.time(); pool.run()
        assert time.time() - t0 < 5


# ---------------------------------------------------------------------------
# PlannerWorkerSwarm philosophy (mocked LLM)
# ---------------------------------------------------------------------------
class TestPhilosophy:
    def test_planner_produces_tasks(self):
        s = _swarm()
        spec = PlannerTaskSpec(plan="My plan", tasks=[
            PlannerTaskOutput(title="T1", description="D1", priority=1),
            PlannerTaskOutput(title="T2", description="D2", priority=2, depends_on_titles=["T1"]),
        ])
        with patch.object(Agent, "run", return_value=spec):
            tasks = s._run_planner("Build feature")
        assert len(tasks) == 2 and len(s.task_queue) == 2
        assert tasks[0].id in s.task_queue.get_task(tasks[1].id).depends_on

    def test_fresh_start_clears_all(self):
        s = _swarm()
        s.task_queue.add_tasks([PlannerTask(title="A", description="d"), PlannerTask(title="B", description="d")])
        _drain(s.task_queue, None)
        s._prepare_next_cycle(CycleVerdict(
            is_complete=False, overall_quality=1, summary="drift", needs_fresh_start=True))
        assert len(s.task_queue) == 0

    def test_gap_fill_preserves_completed(self):
        s = _swarm()
        s.task_queue.add_tasks([PlannerTask(title="A", description="d"), PlannerTask(title="B", description="d")])
        _drain(s.task_queue, None)
        s._prepare_next_cycle(CycleVerdict(
            is_complete=False, overall_quality=5, summary="partial", needs_fresh_start=False))
        assert len(s.task_queue) == 1
        assert s.task_queue.get_all_tasks()[0].status == PlannerTaskStatus.COMPLETED

    def test_full_cycle(self):
        s = _swarm(); seq = [0]
        spec = PlannerTaskSpec(plan="P", tasks=[
            PlannerTaskOutput(title="T1", description="D", priority=1),
            PlannerTaskOutput(title="T2", description="D", priority=1),
        ])
        verdict = CycleVerdict(is_complete=True, overall_quality=9, summary="done")
        def fx(task=None, **kw):
            seq[0] += 1
            if seq[0] == 1: return spec
            if seq[0] <= 3: return f"r{seq[0]}"
            return verdict
        with patch.object(Agent, "run", side_effect=fx):
            assert s.run(task="Goal") is not None
        assert s.task_queue.get_completed_count() == 2

    def test_multi_cycle_feedback(self):
        s = _swarm(max_loops=3); seq = [0]
        s1 = PlannerTaskSpec(plan="P1", tasks=[PlannerTaskOutput(title="T1", description="D", priority=1)])
        s2 = PlannerTaskSpec(plan="P2", tasks=[PlannerTaskOutput(title="T2", description="D", priority=2)])
        v_no = CycleVerdict(is_complete=False, overall_quality=4, summary="gap",
                            gaps=["missing"], follow_up_instructions="add more")
        v_ok = CycleVerdict(is_complete=True, overall_quality=9, summary="done")
        def fx(task=None, **kw):
            seq[0] += 1
            return [s1, "r1", v_no, s2, "r2", v_ok][seq[0]-1]
        with patch.object(Agent, "run", side_effect=fx):
            s.run(task="Goal")
        assert seq[0] == 6

    def test_fresh_start_cycle(self):
        s = _swarm(max_loops=3); seq = [0]
        s1 = PlannerTaskSpec(plan="bad", tasks=[PlannerTaskOutput(title="Drift", description="d", priority=1)])
        s2 = PlannerTaskSpec(plan="fresh", tasks=[PlannerTaskOutput(title="Fresh", description="d", priority=1)])
        v_fresh = CycleVerdict(is_complete=False, overall_quality=1, summary="drift", needs_fresh_start=True,
                               follow_up_instructions="start over", gaps=["all"])
        v_ok = CycleVerdict(is_complete=True, overall_quality=8, summary="ok")
        def fx(task=None, **kw):
            seq[0] += 1
            return [s1, "dr", v_fresh, s2, "fr", v_ok][seq[0]-1]
        with patch.object(Agent, "run", side_effect=fx):
            s.run(task="Test")
        titles = {t.title for t in s.task_queue.get_all_tasks()}
        assert "Drift" not in titles and "Fresh" in titles

    def test_sub_planner_decomposes_critical(self):
        s = _swarm(max_planner_depth=2); seq = [0]
        top = PlannerTaskSpec(plan="top", tasks=[
            PlannerTaskOutput(title="Crit", description="big", priority=3),
            PlannerTaskOutput(title="Normal", description="small", priority=1),
        ])
        sub = PlannerTaskSpec(plan="sub", tasks=[
            PlannerTaskOutput(title="Sub1", description="p1", priority=2),
            PlannerTaskOutput(title="Sub2", description="p2", priority=2),
        ])
        def fx(task=None, **kw):
            seq[0] += 1
            return top if seq[0] == 1 else sub
        with patch.object(Agent, "run", side_effect=fx):
            s._run_planner("test")
        titles = {t.title for t in s.task_queue.get_all_tasks()}
        assert {"Normal", "Sub1", "Sub2"} <= titles


# ---------------------------------------------------------------------------
# Validation & schemas
# ---------------------------------------------------------------------------
class TestValidation:
    def test_no_agents(self):
        with pytest.raises(ValueError): PlannerWorkerSwarm(agents=[])

    def test_bad_max_loops(self):
        with pytest.raises(ValueError): PlannerWorkerSwarm(agents=[_agent()], max_loops=0)

    def test_run_no_task(self):
        with pytest.raises(ValueError): _swarm().run(task=None)

    def test_get_status(self):
        s = _swarm()
        assert s.get_status()["queue"]["total"] == 0

class TestSchemas:
    def test_verdict_fresh_start_default(self):
        assert CycleVerdict(is_complete=True, overall_quality=10, summary="ok").needs_fresh_start is False

    def test_task_defaults(self):
        t = PlannerTask(title="T", description="D")
        assert t.status == PlannerTaskStatus.PENDING and t.id.startswith("ptask-")

    def test_priority_ordering(self):
        assert TaskPriority.LOW < TaskPriority.NORMAL < TaskPriority.HIGH < TaskPriority.CRITICAL


# ---------------------------------------------------------------------------
# Live tests (require API key)
# ---------------------------------------------------------------------------
class TestLive:
    @pytest.mark.live
    def test_end_to_end(self):
        s = PlannerWorkerSwarm(
            agents=[_agent("Research"), _agent("Analysis")],
            max_loops=1, worker_timeout=120,
        )
        assert s.run("What are the top 3 benefits of renewable energy?") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
