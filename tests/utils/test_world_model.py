"""Tests for swarms.utils.world_model."""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from swarms.utils.world_model import (
    AgentWorldModel,
    GraphDelta,
    LinearLatentDynamics,
    SwarmWorldModel,
    WMEdge,
    WMNode,
    attach_shared_swarm_wm,
    extract_graph_delta,
    graph_delta_from_autonomous_subtasks,
    merge_graph_deltas,
    normalize_pearl_relation,
    sanitize_pearl_epistemics,
    truncate_text,
)


def test_truncate_text() -> None:
    assert truncate_text("abc", 10) == "abc"
    assert truncate_text("abcdefghij", 5) == "ab..."


def test_graph_delta_from_autonomous_subtasks_empty() -> None:
    d = graph_delta_from_autonomous_subtasks(
        agent_name="A",
        subtasks=[],
        episode_id="e1",
    )
    assert d.nodes == []
    assert d.edges == []


def test_graph_delta_from_autonomous_subtasks_plan_and_deps() -> None:
    d = graph_delta_from_autonomous_subtasks(
        agent_name="Research Lead",
        subtasks=[
            {
                "step_id": "s1",
                "description": "First",
                "status": "pending",
                "dependencies": [],
            },
            {
                "step_id": "s2",
                "description": "Second",
                "status": "completed",
                "dependencies": ["s1"],
            },
        ],
        plan_task_label="Build AGI",
        episode_id="abc123",
    )
    kinds = {n.id: n.kind for n in d.nodes}
    plan_ids = [nid for nid, k in kinds.items() if k == "plan"]
    assert len(plan_ids) == 1
    st_nodes = [n for n in d.nodes if n.kind == "subtask"]
    assert len(st_nodes) == 2
    assert any("[completed]" in n.label for n in st_nodes)
    rels = {(e.src, e.dst, e.relation) for e in d.edges}
    s1 = next(n.id for n in d.nodes if n.kind == "subtask" and "First" in n.label)
    s2 = next(n.id for n in d.nodes if n.kind == "subtask" and "Second" in n.label)
    assert (s1, s2, "temporal_succession") in rels


def test_merge_graph_deltas_concat() -> None:
    a = GraphDelta(
        nodes=[WMNode(id="x", label="X", kind="fact")],
        edges=[],
    )
    b = graph_delta_from_autonomous_subtasks(
        agent_name="a",
        subtasks=[
            {"step_id": "only", "description": "d", "status": "pending", "dependencies": []}
        ],
        episode_id="z",
    )
    m = merge_graph_deltas(a, b)
    assert len(m.nodes) == 1 + len(b.nodes)
    assert len(m.edges) == len(b.edges)


def test_graph_delta_validation() -> None:
    d = GraphDelta(
        nodes=[WMNode(id="a", label="A", kind="fact")],
        edges=[WMEdge(src="a", dst="b", relation="next")],
    )
    assert len(d.nodes) == 1
    assert d.nodes[0].scope == "semantic"
    assert d.edges[0].relation == "next"


def test_wm_node_scope_coercion() -> None:
    n = WMNode.model_validate(
        {
            "id": "x",
            "label": "L",
            "kind": "observation",
            "scope": "EPISODIC",
        }
    )
    assert n.scope == "episodic"
    n2 = WMNode.model_validate(
        {"id": "y", "label": "M", "kind": "fact", "scope": "bogus"}
    )
    assert n2.scope == "semantic"


def test_apply_delta_and_digest() -> None:
    g = AgentWorldModel("test-agent")
    delta = GraphDelta(
        nodes=[
            WMNode(id="n1", label="User asked X", kind="fact"),
            WMNode(id="n2", label="Answer Y", kind="outcome"),
        ],
        edges=[WMEdge(src="n1", dst="n2", relation="effect")],
    )
    g.apply_delta(
        delta,
        source_tag="t1",
        wm_loop=2,
        wm_episode="ep99",
    )
    d = g.digest_for_hook()
    assert "nodes=2" in d
    assert "edges=1" in d
    # Pearl hygiene: "effect" overclaim -> hypothesized_influence
    assert "hypothesized_influence" in d
    assert "effect" not in d.split("top_relations=")[-1]
    raw = g.to_dict()
    n1 = next(x for x in raw["nodes"] if x["node_id"] == "n1")
    assert n1["wm_loop"] == 2
    assert n1["wm_episode"] == "ep99"
    assert n1["scope"] == "semantic"


def test_pearl_normalization() -> None:
    assert normalize_pearl_relation("cause") == "hypothesized_influence"
    assert normalize_pearl_relation("next") == "temporal_succession"
    assert normalize_pearl_relation("supports") == "evidence_supports"
    assert normalize_pearl_relation("unknown_xyz") == "association"


def test_sanitize_pearl_epistemics_idempotent() -> None:
    d = GraphDelta(
        nodes=[WMNode(id="a", label="A", kind="fact")],
        edges=[WMEdge(src="a", dst="b", relation="hypothesized_influence")],
    )
    d2 = sanitize_pearl_epistemics(d)
    assert d2.edges[0].relation == "hypothesized_influence"


def test_to_dict_from_dict_roundtrip() -> None:
    g = AgentWorldModel("a1")
    g.apply_delta(
        GraphDelta(
            nodes=[WMNode(id="x", label="lx", kind="fact")],
            edges=[],
        ),
        "run",
    )
    data = g.to_dict()
    g2 = AgentWorldModel.from_dict(data)
    assert g2.digest_for_hook() == g.digest_for_hook()


def test_linear_latent_dynamics_lstsq_recovery() -> None:
    rng = np.random.default_rng(42)
    latent_dim, action_dim = 3, 2
    n = 80
    Theta_true = rng.standard_normal((latent_dim + action_dim + 1, latent_dim))
    Z = rng.standard_normal((n, latent_dim))
    A = rng.standard_normal((n, action_dim))
    ones = np.ones((n, 1))
    X = np.hstack([Z, A, ones])
    Zn = X @ Theta_true

    eye = np.eye(latent_dim)
    dyn = LinearLatentDynamics(
        obs_dim=latent_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        projection=eye,
    )
    for i in range(n):
        dyn.observe(Z[i], A[i], Zn[i])
    assert dyn.fit() is True
    assert dyn.is_fitted
    assert dyn._Theta is not None
    err = np.linalg.norm(dyn._Theta - Theta_true)
    assert err < 1e-9


def test_linear_latent_dynamics_predict_and_dict_roundtrip() -> None:
    latent_dim, action_dim = 2, 1
    dyn = LinearLatentDynamics(
        obs_dim=latent_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        projection=np.eye(latent_dim),
    )
    for t in range(20):
        o = np.array([float(t), float(t + 1)])
        a = np.array([0.5])
        o2 = o + 0.1 * a[0]
        dyn.observe(o, a, o2)
    assert dyn.fit() is True
    z = dyn.encode(np.array([0.0, 1.0]))
    z_hat = dyn.predict_next_latent(z, np.array([0.5]))
    assert z_hat.shape == (latent_dim,)

    dyn2 = LinearLatentDynamics.from_dict(dyn.to_dict())
    assert dyn2.is_fitted
    assert np.allclose(
        dyn.predict_next_latent(z, np.array([0.5])),
        dyn2.predict_next_latent(z, np.array([0.5])),
    )


def test_agent_world_model_trajectory_in_to_dict() -> None:
    g = AgentWorldModel("wm")
    g.init_trajectory_dynamics(2, 1, 2, projection=np.eye(2))
    assert g.trajectory_dynamics is not None
    td = g.trajectory_dynamics
    for i in range(6):
        td.observe([float(i), float(i + 1)], [0.25], [float(i) + 0.1, float(i + 1) + 0.1])
    assert td.fit() is True
    raw = g.to_dict()
    assert "trajectory_dynamics" in raw
    g2 = AgentWorldModel.from_dict(raw)
    assert g2.trajectory_dynamics is not None
    assert g2.trajectory_dynamics.is_fitted


def test_swarm_world_model_from_dict() -> None:
    g = SwarmWorldModel("sw")
    g.apply_delta(
        GraphDelta(
            nodes=[WMNode(id="s", label="shared", kind="fact")],
            edges=[],
        ),
        "a",
    )
    g2 = SwarmWorldModel.from_dict(g.to_dict())
    assert "nodes=1" in g2.digest_for_hook()


def test_extract_graph_delta_mocked() -> None:
    payload = {
        "nodes": [{"id": "a", "label": "task", "kind": "fact"}],
        "edges": [],
    }
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps(payload)))
    ]
    with patch("swarms.utils.world_model.litellm.completion", return_value=mock_resp):
        delta = extract_graph_delta(
            task="hello",
            assistant_reply="world",
            model="gpt-4.1",
        )
    assert len(delta.nodes) == 1
    assert delta.nodes[0].id == "a"


def test_extract_graph_delta_invalid_json_fallback() -> None:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="not json"))]
    with patch("swarms.utils.world_model.litellm.completion", return_value=mock_resp):
        delta = extract_graph_delta(
            task="t",
            assistant_reply="r",
            model="gpt-4.1",
        )
    assert len(delta.nodes) == 1
    assert delta.nodes[0].kind == "unparsed"
    assert delta.nodes[0].scope == "episodic"


def test_extract_graph_delta_scope_in_payload() -> None:
    payload = {
        "nodes": [
            {
                "id": "u_task",
                "label": "User asked",
                "kind": "observation",
                "scope": "episodic",
            }
        ],
        "edges": [],
    }
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps(payload)))
    ]
    with patch("swarms.utils.world_model.litellm.completion", return_value=mock_resp):
        delta = extract_graph_delta(
            task="hello",
            assistant_reply="world",
            model="gpt-4.1",
        )
    assert delta.nodes[0].scope == "episodic"
    assert delta.nodes[0].kind == "observation"


class _FakeAgent:
    def __init__(self, wm: bool = True, swarm_wm=None) -> None:
        self.wm = wm
        self.swarm_wm = swarm_wm


def test_attach_shared_swarm_wm() -> None:
    shared = SwarmWorldModel("team")
    a1 = _FakeAgent(wm=True, swarm_wm=None)
    a2 = _FakeAgent(wm=True, swarm_wm=None)
    a3 = _FakeAgent(wm=False, swarm_wm=None)
    attach_shared_swarm_wm([a1, a2, a3], shared)
    assert a1.swarm_wm is shared
    assert a2.swarm_wm is shared
    assert a3.swarm_wm is None
