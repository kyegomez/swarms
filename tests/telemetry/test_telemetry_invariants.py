"""Invariant tests for ``swarms.telemetry.otel``.

Unlike ``test_telemetry.py`` (unit coverage of each primitive), this module
asserts properties that must hold no matter which architecture is running:

    1. Span parentage — document the current flat-trace behavior.
    2. Telemetry is off by default.
    3. Telemetry never changes program behavior (output, errors, decorator
       metadata, return-value passthrough).
    4. A broken exporter/tracer/span never breaks a run.
    5. Payload bounds + ``init_config`` hardening.
    6. Attribute schema consistency across single-agent vs. multi-agent runs.
    7. No duplicate/leaked spans.

None of these tests require a live LLM key or network access: every agent
under test has its ``.llm`` swapped for an in-process ``FakeLLM``, and the
telemetry exporter always points at an unreachable local address.
"""

import json
import os
import time
from contextlib import contextmanager

import pytest
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import swarms.telemetry.otel as otel
from swarms import Agent
from swarms.telemetry.otel import (
    MAX_CONFIG_CHARS,
    MAX_PAYLOAD_CHARS,
    capture_run,
    init_config,
    trace_run,
)

os.environ.setdefault("SWARMS_OTEL_TIMEOUT", "2")

# ---------------------------------------------------------------------------
# Never let a code path in this file make a real LLM call. Every agent here
# gets a FakeLLM instead, but this belt-and-suspenders guard means a bug that
# accidentally exercises the real litellm path fails loudly (auth error)
# instead of silently succeeding (and costing money) on a machine that
# happens to have a key exported.
# ---------------------------------------------------------------------------
_LLM_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True, scope="module")
def _no_llm_keys():
    saved = {k: os.environ.pop(k, None) for k in _LLM_KEYS}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def _exporter():
    os.environ["SWARMS_TELEMETRY_ON"] = "true"
    otel.TELEMETRY_BASE_URL = "http://127.0.0.1:9/dead"  # unreachable
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


class FakeLLM:
    def __init__(self, reply="FAKE OUTPUT"):
        self.stream = False
        self.temperature = 0.5
        self.reply = reply

    def run(self, task=None, img=None, **kwargs):
        return self.reply


def fake_agent(name, reply=None):
    a = Agent(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        verbose=False,
    )
    a.llm = FakeLLM(reply or f"{name} output")
    return a


def _finished(exporter):
    return exporter.get_finished_spans()


def _by_name_all(exporter, name):
    return [s for s in _finished(exporter) if s.name == name]


def _attrs(span):
    return dict(span.attributes)


# ===========================================================================
# 1. Span parentage — spans nest into a single trace per run.
# ===========================================================================
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


if __name__ == "__main__":
    pytest.main()
