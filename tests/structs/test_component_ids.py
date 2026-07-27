"""
Regression tests for per-instance component IDs.

Every ``Agent`` and swarm used to share a single ``id`` for the lifetime of the
process. The constructors declared ``id: str = agent_id()`` / ``swarm_id()``,
and Python evaluates a default argument **once, at import time** — so the
generated string was baked into the function signature and handed to every
instance that did not pass one explicitly.

The consequence reached well past cosmetics: telemetry spans carry ``swarms.id``,
so an agent's ``init`` span (which holds its system prompt and full config) could
not be joined to its ``run`` spans, and two agents in the same trace were
indistinguishable. Anything else keyed on ID — registries, dedup, per-agent
caches, persisted state — was equally unsafe.

The fix defaults ``id`` to ``None`` and generates inside ``__init__``.

Run:
    pytest tests/structs/test_component_ids.py -v
"""

import inspect

import pytest

from swarms import (
    Agent,
    AgentRearrange,
    CouncilAsAJudge,
    GraphWorkflow,
    MajorityVoting,
    MixtureOfAgents,
    MultiAgentRouter,
    SwarmRouter,
)
from swarms.agents.agent_judge import AgentJudge
from swarms.agents.reasoning_duo import ReasoningDuo
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow
from swarms.structs.conversation import Conversation
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.swarm_rearrange import SwarmRearrange


########################################################
# Helpers
########################################################


def _agent(name: str = "IdTest") -> Agent:
    """An offline agent — constructed only, never run."""
    return Agent(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        verbose=False,
    )


class _FakeSwarm:
    """Minimal stand-in for SwarmRearrange's swarm list."""

    def __init__(self, name: str = "s1"):
        self.name = name

    def run(self, *args, **kwargs):
        return "ok"


# Each entry builds a fresh instance with no explicit id.
BUILDERS = {
    "Agent": lambda: _agent(),
    "AgentRearrange": lambda: AgentRearrange(
        agents=[_agent("ar")], flow="ar"
    ),
    "MajorityVoting": lambda: MajorityVoting(agents=[_agent("mv")]),
    "BatchedGridWorkflow": lambda: BatchedGridWorkflow(
        agents=[_agent("bg")]
    ),
    "CouncilAsAJudge": lambda: CouncilAsAJudge(),
    "SwarmRearrange": lambda: SwarmRearrange(
        swarms=[_FakeSwarm()], flow="s1"
    ),
    "LLMCouncil": lambda: LLMCouncil(),
    "Conversation": lambda: Conversation(),
    "GraphWorkflow": lambda: GraphWorkflow(),
    "AgentJudge": lambda: AgentJudge(),
    "ReasoningDuo": lambda: ReasoningDuo(),
    "SwarmRouter": lambda: SwarmRouter(
        name="IdRouter", agents=[_agent("sr")]
    ),
    "MixtureOfAgents": lambda: MixtureOfAgents(
        agents=[_agent("moa")], aggregator_agent=_agent("agg")
    ),
    "MultiAgentRouter": lambda: MultiAgentRouter(
        agents=[_agent("mar")]
    ),
}

CLASSES = {
    "Agent": Agent,
    "AgentRearrange": AgentRearrange,
    "MajorityVoting": MajorityVoting,
    "BatchedGridWorkflow": BatchedGridWorkflow,
    "CouncilAsAJudge": CouncilAsAJudge,
    "SwarmRearrange": SwarmRearrange,
    "LLMCouncil": LLMCouncil,
    "Conversation": Conversation,
    "GraphWorkflow": GraphWorkflow,
    "AgentJudge": AgentJudge,
    "ReasoningDuo": ReasoningDuo,
    "SwarmRouter": SwarmRouter,
    "MixtureOfAgents": MixtureOfAgents,
    "MultiAgentRouter": MultiAgentRouter,
}


########################################################
# The defaults themselves
########################################################


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_id_default_is_none_not_a_frozen_string(name):
    """The signature default must be None.

    If a generator is called in the signature, the default becomes a concrete
    string — which is exactly the bug. Reading it off the signature catches a
    regression without constructing anything.
    """
    default = (
        inspect.signature(CLASSES[name].__init__)
        .parameters["id"]
        .default
    )

    assert default is None, (
        f"{name}.__init__ has a pre-computed id default ({default!r}). "
        "Default arguments are evaluated once at import, so every instance "
        "would share this id. Use `id: Optional[str] = None` and generate "
        "inside __init__."
    )


########################################################
# Behavior
########################################################


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_ids_are_unique_per_instance(name):
    first = BUILDERS[name]().id
    second = BUILDERS[name]().id

    assert first and second, f"{name} produced an empty id"
    assert (
        first != second
    ), f"two {name} instances share the id {first!r}"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_id_is_populated(name):
    """Every component exposes a non-empty string id.

    LLMCouncil previously accepted `id` and never stored it, so it had no
    `.id` attribute at all — telemetry recorded nothing for it.
    """
    instance = BUILDERS[name]()

    assert hasattr(instance, "id"), f"{name} has no id attribute"
    assert isinstance(instance.id, str)
    assert instance.id.strip()


def test_agent_explicit_id():
    assert (
        Agent(
            id="fixed-agent-id",
            agent_name="Explicit",
            model_name="gpt-4o-mini",
            persistent_memory=False,
            print_on=False,
        ).id
        == "fixed-agent-id"
    )


def test_swarm_explicit_id():
    assert (
        MajorityVoting(id="fixed-swarm-id", agents=[_agent()]).id
        == "fixed-swarm-id"
    )


########################################################
# Prefixes stay intact
########################################################


@pytest.mark.parametrize(
    "name,prefix",
    [
        ("Agent", "agent-"),
        ("AgentRearrange", "agent-rearrange-"),
        ("MajorityVoting", "majority-voting-"),
        ("BatchedGridWorkflow", "batched-grid-workflow-"),
        ("CouncilAsAJudge", "council-as-judge-"),
        ("SwarmRearrange", "swarm-rearrange-"),
        ("LLMCouncil", "llm-council-"),
        ("GraphWorkflow", "graph-workflow-"),
        ("AgentJudge", "agent-judge-"),
        ("ReasoningDuo", "reasoning-duo-"),
        ("SwarmRouter", "swarm-router-"),
        ("MixtureOfAgents", "mixture-of-agents-"),
        ("MultiAgentRouter", "multi-agent-router-"),
    ],
)
def test_ids_are_prefixed_by_component(name, prefix):
    """Every id is self-describing, so it identifies its own kind wherever it
    surfaces — logs, telemetry spans, persisted state."""
    assert BUILDERS[name]().id.startswith(prefix)


########################################################
# Many instances, one process
########################################################


def test_fifty_agents_have_fifty_distinct_ids():
    ids = {_agent(f"Bulk-{i}").id for i in range(50)}
    assert len(ids) == 50


########################################################
# No constructor may compute a default by calling something
########################################################


def test_no_constructor_computes_a_default_argument():
    """A package-wide guard against the whole class of bug.

    Any `def __init__(self, x=f())` bakes one value into the signature at
    import time and hands it to every instance. Catching it structurally means
    a new component cannot reintroduce the problem unnoticed.
    """
    import ast
    import pathlib

    offenders = []
    root = pathlib.Path(__file__).resolve().parents[2] / "swarms"

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue

        for cls in [
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        ]:
            init = next(
                (
                    m
                    for m in cls.body
                    if isinstance(m, ast.FunctionDef)
                    and m.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue

            positional = init.args.args[
                len(init.args.args) - len(init.args.defaults) :
            ]
            for arg, default in zip(positional, init.args.defaults):
                if isinstance(default, ast.Call):
                    offenders.append(
                        f"{path.relative_to(root.parent)}:{default.lineno} "
                        f"{cls.name}.__init__({arg.arg}={ast.unparse(default)})"
                    )

    assert not offenders, (
        "constructor defaults are evaluated once at import, so these are "
        "shared by every instance:\n  " + "\n  ".join(offenders)
    )


########################################################
# One generator for the whole framework
########################################################


def test_every_component_id_comes_from_generate_id():
    """No component may hand-roll its own id generator.

    Before consolidation there were four — ``agent_id()``, ``swarm_id()``,
    ``generate_conversation_id()`` and ``generate_api_key(prefix=...)`` — plus
    inline ``str(uuid.uuid4())`` calls, so id formats drifted per component.
    """
    import ast
    import pathlib

    offenders = []
    root = pathlib.Path(__file__).resolve().parents[2] / "swarms"

    for path in sorted(root.rglob("*.py")):
        if (
            "__pycache__" in str(path)
            or path.name == "generate_id.py"
        ):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Attribute) and t.attr == "id"
                    for t in node.targets
                )
            ):
                continue

            rendered = ast.unparse(node.value)
            if "uuid" in rendered or "generate_api_key" in rendered:
                offenders.append(
                    f"{path.relative_to(root.parent)}:{node.lineno} {rendered}"
                )

    assert not offenders, (
        "these assign an id without going through generate_id():\n  "
        + "\n  ".join(offenders)
    )


def test_generate_id_shape():
    from swarms.utils.generate_id import generate_id

    prefixed = generate_id("agent")
    assert prefixed.startswith("agent-")
    assert len(prefixed.split("-", 1)[1]) == 32

    bare = generate_id()
    assert "-" not in bare
    assert len(bare) == 32

    assert generate_id("x") != generate_id("x")
