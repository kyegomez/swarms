"""
Cognitive world model utilities for Swarms agents.

This module provides a lightweight **epistemic graph** (NetworkX DiGraph) that
summarizes task / assistant turns into nodes and edges. Edge labels follow a
**conservative** reading aligned with observational vs causal distinction
(Judea Pearl): we **never** store bare ``cause``/``effect`` as fact---those are
automatically downgraded to *hypothesized* or *associational* relations unless
stronger evidence exists (which plain chat does not provide). The graph is not
a full structural causal model (SCM). Optionally, :class:`LinearLatentDynamics`
fits **linear latent dynamics** from ``(observation, action, next_observation)``
trajectories (fixed random linear encoder + least-squares transition), for
lightweight rollouts in latent space only.

Enable on an agent with ``Agent(..., wm=True)``. Optionally pass a shared
:class:`SwarmWorldModel` as ``swarm_wm=`` so multiple agents merge into one graph.
Use ``on_wm_update`` for hooks (e.g. meta-orchestration) without automatic
prompt mutation.

Extracted nodes carry a ``scope`` (``semantic`` vs ``episodic``) and each merge
can record ``wm_loop`` plus a per-``run()`` ``wm_episode`` id on graph elements
for longitudinal debugging and autonomous workflows.

When an agent uses the autonomous loop, :func:`graph_delta_from_autonomous_subtasks`
builds a deterministic :class:`GraphDelta` from ``autonomous_subtasks`` (plan
steps) and merges it alongside LLM-extracted nodes on each world-model update.

The package barrel :mod:`swarms.utils` re-exports ``AgentWorldModel``,
``SwarmWorldModel``, and ``LinearLatentDynamics``; import other names from this
module directly when needed.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import litellm
import networkx as nx
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# --- Pydantic schemas ---------------------------------------------------------


class WMNode(BaseModel):
    """A single concept or observation in the cognitive graph."""

    id: str = Field(..., min_length=1, description="Stable id within this delta")
    label: str = Field(..., min_length=1, description="Short human-readable text")
    kind: str = Field(
        default="fact",
        description=(
            "e.g. fact, hypothesis, action, outcome, observation (this-turn), unparsed"
        ),
    )
    scope: Literal["semantic", "episodic"] = Field(
        default="semantic",
        description=(
            "semantic=durable claim or definition; episodic=this-turn utterance or step"
        ),
    )

    @field_validator("scope", mode="before")
    @classmethod
    def _coerce_scope(cls, v: Any) -> str:
        if v is None or v == "":
            return "semantic"
        s = str(v).strip().lower()
        if s in ("semantic", "episodic"):
            return s
        return "semantic"


class WMEdge(BaseModel):
    """Directed link between two node ids (relation is epistemically safe label)."""

    src: str = Field(..., min_length=1)
    dst: str = Field(..., min_length=1)
    relation: str = Field(
        ...,
        min_length=1,
        description=(
            "Pearl-style conservative tag: association, hypothesized_influence, "
            "temporal_succession, evidence_supports, evidence_conflicts"
        ),
    )


class GraphDelta(BaseModel):
    """Structured graph update from one extract call."""

    nodes: List[WMNode] = Field(default_factory=list)
    edges: List[WMEdge] = Field(default_factory=list)

    @field_validator("nodes", "edges", mode="before")
    @classmethod
    def _coerce_none(cls, v: Any) -> Any:
        return v if v is not None else []


# --- Trajectory: latent encoder + linear dynamics (numpy) --------------------


_ArrayLike = Union[np.ndarray, Sequence[float]]


class LinearLatentDynamics:
    """
    Trajectory model: fixed linear encoder ``z = P @ o`` and learned affine
    dynamics ``z' ≈ [z, a, 1] @ Θ`` fitted by least squares on buffered tuples.

    The encoder ``P`` is either a supplied matrix or a row-normalized random
    Gaussian draw (Johnson–Lindenstrauss style). This is not a neural world
    model; it is a small, dependency-light baseline for latent one-step prediction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        *,
        seed: int = 0,
        max_buffer: int = 10_000,
        projection: Optional[np.ndarray] = None,
    ) -> None:
        if obs_dim < 1 or action_dim < 1 or latent_dim < 1:
            raise ValueError("obs_dim, action_dim, and latent_dim must be >= 1")
        if max_buffer < 1:
            raise ValueError("max_buffer must be >= 1")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_buffer = max_buffer
        self._lock = threading.Lock()
        self._z: List[np.ndarray] = []
        self._a: List[np.ndarray] = []
        self._z_next: List[np.ndarray] = []
        self._Theta: Optional[np.ndarray] = None

        if projection is not None:
            P = np.asarray(projection, dtype=np.float64)
            if P.shape != (latent_dim, obs_dim):
                raise ValueError(
                    f"projection must have shape ({latent_dim}, {obs_dim}), got {P.shape}"
                )
            self._P = P
        else:
            rng = np.random.default_rng(seed)
            P = rng.standard_normal((latent_dim, obs_dim))
            row_norms = np.linalg.norm(P, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-8)
            self._P = (P / row_norms).astype(np.float64)

    @property
    def is_fitted(self) -> bool:
        return self._Theta is not None

    @property
    def n_samples(self) -> int:
        with self._lock:
            return len(self._z)

    def encode(self, observation: _ArrayLike) -> np.ndarray:
        o = np.asarray(observation, dtype=np.float64).reshape(-1)
        if o.shape[0] != self.obs_dim:
            raise ValueError(
                f"observation must have length {self.obs_dim}, got {o.shape[0]}"
            )
        return self._P @ o

    def observe(
        self,
        observation: _ArrayLike,
        action: _ArrayLike,
        next_observation: _ArrayLike,
    ) -> None:
        z = self.encode(observation)
        z_next = self.encode(next_observation)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(
                f"action must have length {self.action_dim}, got {a.shape[0]}"
            )
        with self._lock:
            if len(self._z) >= self.max_buffer:
                self._z.pop(0)
                self._a.pop(0)
                self._z_next.pop(0)
            self._z.append(z)
            self._a.append(a)
            self._z_next.append(z_next)

    def fit(self) -> bool:
        """
        Fit Θ from buffered transitions. Returns False if there are too few samples.
        """
        with self._lock:
            need = self.latent_dim + self.action_dim + 1
            n = len(self._z)
            if n < need:
                logger.warning(
                    "LinearLatentDynamics.fit: need at least {} samples, have {}",
                    need,
                    n,
                )
                return False
            Z = np.stack(self._z, axis=0)
            A = np.stack(self._a, axis=0)
            Zn = np.stack(self._z_next, axis=0)
            ones = np.ones((Z.shape[0], 1), dtype=np.float64)
            X = np.hstack([Z, A, ones])
            Theta, *_ = np.linalg.lstsq(X, Zn, rcond=None)
            self._Theta = Theta
        logger.debug(
            "LinearLatentDynamics fit ok samples={} latent={} action={}",
            n,
            self.latent_dim,
            self.action_dim,
        )
        return True

    def predict_next_latent(
        self,
        latent: _ArrayLike,
        action: _ArrayLike,
    ) -> np.ndarray:
        if self._Theta is None:
            raise RuntimeError("LinearLatentDynamics: call fit() before predict_next_latent")
        z = np.asarray(latent, dtype=np.float64).reshape(-1)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if z.shape[0] != self.latent_dim:
            raise ValueError(f"latent must have length {self.latent_dim}")
        if a.shape[0] != self.action_dim:
            raise ValueError(f"action must have length {self.action_dim}")
        x = np.concatenate([z, a, np.ones(1, dtype=np.float64)])
        return x @ self._Theta

    def clear_buffer(self) -> None:
        with self._lock:
            self._z.clear()
            self._a.clear()
            self._z_next.clear()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "latent_dim": self.latent_dim,
                "max_buffer": self.max_buffer,
                "P": self._P.tolist(),
                "Theta": None if self._Theta is None else self._Theta.tolist(),
                "n_samples": len(self._z),
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinearLatentDynamics":
        P = np.asarray(data["P"], dtype=np.float64)
        obs_dim = int(data["obs_dim"])
        action_dim = int(data["action_dim"])
        latent_dim = int(data["latent_dim"])
        inst = cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            seed=0,
            max_buffer=int(data.get("max_buffer", 10_000)),
            projection=P,
        )
        if data.get("Theta") is not None:
            inst._Theta = np.asarray(data["Theta"], dtype=np.float64)
        return inst


def _restore_trajectory_from_dict(holder: "_CognitiveGraphMixin", data: Dict[str, Any]) -> None:
    raw = data.get("trajectory_dynamics")
    if isinstance(raw, dict):
        holder.trajectory_dynamics = LinearLatentDynamics.from_dict(raw)


# --- Graph containers ---------------------------------------------------------


class _CognitiveGraphMixin:
    """Shared NetworkX-backed graph with thread-safe mutation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = threading.Lock()
        self.trajectory_dynamics: Optional[LinearLatentDynamics] = None

    def init_trajectory_dynamics(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        *,
        seed: int = 0,
        max_buffer: int = 10_000,
        projection: Optional[np.ndarray] = None,
    ) -> LinearLatentDynamics:
        """
        Attach a :class:`LinearLatentDynamics` model to this graph container.

        Replaces any previous trajectory model on this instance.
        """
        self.trajectory_dynamics = LinearLatentDynamics(
            obs_dim,
            action_dim,
            latent_dim,
            seed=seed,
            max_buffer=max_buffer,
            projection=projection,
        )
        return self.trajectory_dynamics

    def apply_delta(
        self,
        delta: GraphDelta,
        source_tag: str,
        *,
        wm_loop: Optional[int] = None,
        wm_episode: Optional[str] = None,
    ) -> None:
        """
        Merge nodes and edges into the graph.

        Args:
            delta: Validated extract result.
            source_tag: Provenance (e.g. agent name or run id).
            wm_loop: Optional agent internal loop index for this merge.
            wm_episode: Optional id grouping all merges from one ``Agent.run()`` call.
        """
        delta = sanitize_pearl_epistemics(delta)
        merge_extras: Dict[str, Any] = {"source_tag": source_tag}
        if wm_loop is not None:
            merge_extras["wm_loop"] = wm_loop
        if wm_episode is not None:
            merge_extras["wm_episode"] = wm_episode
        with self._lock:
            for n in delta.nodes:
                self._graph.add_node(
                    n.id,
                    label=n.label,
                    kind=n.kind,
                    scope=n.scope,
                    **merge_extras,
                )
            for e in delta.edges:
                if e.src not in self._graph:
                    self._graph.add_node(
                        e.src,
                        label=e.src,
                        kind="implicit",
                        scope="episodic",
                        **merge_extras,
                    )
                if e.dst not in self._graph:
                    self._graph.add_node(
                        e.dst,
                        label=e.dst,
                        kind="implicit",
                        scope="episodic",
                        **merge_extras,
                    )
                self._graph.add_edge(
                    e.src,
                    e.dst,
                    relation=e.relation,
                    **merge_extras,
                )

    def digest_for_hook(self) -> str:
        """Compact summary for callbacks (no full text)."""
        with self._lock:
            nn = self._graph.number_of_nodes()
            ne = self._graph.number_of_edges()
            rel_counts: Dict[str, int] = {}
            for _, _, data in self._graph.edges(data=True):
                r = str(data.get("relation", "?"))
                rel_counts[r] = rel_counts.get(r, 0) + 1
            parts = [f"name={self.name}", f"nodes={nn}", f"edges={ne}"]
            if rel_counts:
                top = sorted(rel_counts.items(), key=lambda x: -x[1])[:6]
                parts.append(
                    "top_relations=" + ",".join(f"{k}:{v}" for k, v in top)
                )
            return "; ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize nodes and edges for optional persistence."""
        with self._lock:
            nodes_data = [
                dict(node_id=n, **self._graph.nodes[n]) for n in self._graph.nodes()
            ]
            edges_data = [
                {"src": u, "dst": v, **data}
                for u, v, data in self._graph.edges(data=True)
            ]
            out: Dict[str, Any] = {
                "name": self.name,
                "nodes": nodes_data,
                "edges": edges_data,
            }
            if self.trajectory_dynamics is not None:
                out["trajectory_dynamics"] = self.trajectory_dynamics.to_dict()
            return out


def _populate_graph_from_dict(graph: nx.DiGraph, data: Dict[str, Any]) -> None:
    graph.clear()
    for n in data.get("nodes", []):
        if not isinstance(n, dict) or "node_id" not in n:
            continue
        nid = n["node_id"]
        attrs = {k: v for k, v in n.items() if k != "node_id"}
        graph.add_node(nid, **attrs)
    for e in data.get("edges", []):
        if not isinstance(e, dict):
            continue
        u, v = e.get("src"), e.get("dst")
        if u is None or v is None:
            continue
        rest = {k: v2 for k, v2 in e.items() if k not in ("src", "dst")}
        graph.add_edge(u, v, **rest)


class AgentWorldModel(_CognitiveGraphMixin):
    """Per-agent cognitive graph."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(name=agent_name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentWorldModel:
        """Restore from :meth:`to_dict` output."""
        inst = cls(agent_name=str(data.get("name", "restored")))
        with inst._lock:
            _populate_graph_from_dict(inst._graph, data)
            _restore_trajectory_from_dict(inst, data)
        return inst


class SwarmWorldModel(_CognitiveGraphMixin):
    """Optional shared graph for multiple agents."""

    def __init__(self, name: str = "swarm") -> None:
        super().__init__(name=name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SwarmWorldModel:
        """Restore from :meth:`to_dict` output."""
        inst = cls(name=str(data.get("name", "swarm")))
        with inst._lock:
            _populate_graph_from_dict(inst._graph, data)
            _restore_trajectory_from_dict(inst, data)
        return inst


# --- Helpers ------------------------------------------------------------------


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _wm_slug(s: str, max_len: int = 40) -> str:
    """Stable ASCII-ish token for world-model node ids (avoids NetworkX quirks)."""
    t = re.sub(r"[^\w\-]+", "_", str(s).strip(), flags=re.UNICODE)
    t = re.sub(r"_+", "_", t).strip("_").lower()
    if not t:
        t = "x"
    return t[:max_len]


def graph_delta_from_autonomous_subtasks(
    *,
    agent_name: str,
    subtasks: List[Dict[str, Any]],
    plan_task_label: str = "",
    episode_id: Optional[str] = None,
) -> GraphDelta:
    """
    Build a :class:`GraphDelta` from the agent's ``autonomous_subtasks`` list.

    Creates a ``kind="plan"`` root node, one ``kind="subtask"`` node per step,
    edges from plan to each subtask (``association``), and dependency edges between
    subtasks (``temporal_succession`` from prerequisite to dependent).

    Idempotent merges: node ids are namespaced by agent slug and optional
    ``episode_id`` (per ``Agent.run()``) so shared swarm graphs stay disjoint
    across agents and runs.
    """
    if not subtasks:
        return GraphDelta()
    ag = _wm_slug(agent_name or "agent", 32)
    ep = _wm_slug(episode_id, 24) if episode_id else "noep"
    plan_id = f"wm_plan_{ag}_{ep}"

    plan_label = (
        truncate_text(plan_task_label.strip(), 120)
        if plan_task_label and str(plan_task_label).strip()
        else truncate_text(f"Plan — {agent_name or 'agent'}", 120)
    )
    nodes: List[WMNode] = [
        WMNode(
            id=plan_id,
            label=plan_label,
            kind="plan",
            scope="episodic",
        )
    ]
    edges: List[WMEdge] = []

    step_to_nid: Dict[str, str] = {}
    for st in subtasks:
        if not isinstance(st, dict):
            continue
        sid = str(st.get("step_id", "")).strip()
        if not sid:
            continue
        nid = f"wm_st_{ag}_{ep}_{_wm_slug(sid)}"
        step_to_nid[sid] = nid
        desc = str(st.get("description", ""))
        status = str(st.get("status", "pending"))
        label = truncate_text(f"[{status}] {desc}", 120)
        nodes.append(
            WMNode(
                id=nid,
                label=label,
                kind="subtask",
                scope="episodic",
            )
        )
        edges.append(WMEdge(src=plan_id, dst=nid, relation="association"))

    for st in subtasks:
        if not isinstance(st, dict):
            continue
        sid = str(st.get("step_id", "")).strip()
        dst = step_to_nid.get(sid)
        if not dst:
            continue
        for dep in st.get("dependencies") or []:
            dep_s = str(dep).strip()
            src = step_to_nid.get(dep_s)
            if src and src != dst:
                edges.append(
                    WMEdge(src=src, dst=dst, relation="temporal_succession")
                )

    return GraphDelta(nodes=nodes, edges=edges)


def merge_graph_deltas(*parts: GraphDelta) -> GraphDelta:
    """Concatenate graph deltas (later duplicates overwrite on :meth:`~_CognitiveGraphMixin.apply_delta`)."""
    nodes: List[WMNode] = []
    edges: List[WMEdge] = []
    for d in parts:
        nodes.extend(d.nodes)
        edges.extend(d.edges)
    return GraphDelta(nodes=nodes, edges=edges)


def _strip_json_fence(content: str) -> str:
    s = content.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


# --- Pearl-style epistemic hygiene (automatic; no user config) ----------------
# Closed vocabulary: text-only evidence cannot justify structural causal claims.

_ALLOWED_RELATIONS: frozenset[str] = frozenset(
    {
        "association",
        "hypothesized_influence",
        "temporal_succession",
        "evidence_supports",
        "evidence_conflicts",
    }
)

# Map LLM output (and legacy labels) -> allowed relation (lowercase keys).
_RELATION_ALIASES: Dict[str, str] = {
    # Causal overclaim -> hypothetical influence (Pearl: not identified from text)
    "cause": "hypothesized_influence",
    "causes": "hypothesized_influence",
    "causal": "hypothesized_influence",
    "causation": "hypothesized_influence",
    "because": "hypothesized_influence",
    "effect": "hypothesized_influence",
    "effects": "hypothesized_influence",
    "result": "hypothesized_influence",
    "results": "hypothesized_influence",
    "results_in": "hypothesized_influence",
    "leads_to": "hypothesized_influence",
    "leads to": "hypothesized_influence",
    "reaction": "association",
    "reacts": "association",
    "response": "association",
    # Pure order (Pearl ladder: association / time without causal ID)
    "next": "temporal_succession",
    "then": "temporal_succession",
    "before": "temporal_succession",
    "after": "temporal_succession",
    "follows": "temporal_succession",
    "precedes": "temporal_succession",
    # Evidence about claims
    "supports": "evidence_supports",
    "support": "evidence_supports",
    "contradicts": "evidence_conflicts",
    "contradict": "evidence_conflicts",
    "against": "evidence_conflicts",
    # Associational
    "correlates": "association",
    "correlation": "association",
    "associated": "association",
    "cooccurs": "association",
    "mentions": "association",
    "describes": "association",
    "related": "association",
}


def normalize_pearl_relation(relation: str) -> str:
    """
    Map any free-text relation to the closed, epistemically conservative set.

    Callers normally use :func:`sanitize_pearl_epistemics`; this is public for tests.
    """
    key = relation.strip().lower().replace(" ", "_").replace("-", "_")
    if not key:
        return "association"
    if key in _ALLOWED_RELATIONS:
        return key
    mapped = _RELATION_ALIASES.get(key)
    if mapped is not None:
        return mapped
    # Substring fallbacks for compound junk from models
    if "causal" in key or key.startswith("cause"):
        return "hypothesized_influence"
    if "effect" in key and key not in ("side_effect",):
        return "hypothesized_influence"
    if "next" in key or "temporal" in key or "sequence" in key:
        return "temporal_succession"
    if "support" in key:
        return "evidence_supports"
    if "contradict" in key or "conflict" in key:
        return "evidence_conflicts"
    return "association"


def sanitize_pearl_epistemics(delta: GraphDelta) -> GraphDelta:
    """
    Rewrite edges so nothing looks like proven causation from chat text alone.

    Runs automatically on every merge; users need not configure anything.
    """
    if not delta.edges:
        return delta
    new_edges: List[WMEdge] = []
    for e in delta.edges:
        rel = normalize_pearl_relation(e.relation)
        new_edges.append(WMEdge(src=e.src, dst=e.dst, relation=rel))
    return GraphDelta(nodes=delta.nodes, edges=new_edges)


def _fallback_delta(reason: str) -> GraphDelta:
    uid = str(uuid.uuid4())
    return GraphDelta(
        nodes=[
            WMNode(
                id=uid,
                label=truncate_text(reason, 200),
                kind="unparsed",
                scope="episodic",
            )
        ],
        edges=[],
    )


_EXTRACT_SYSTEM = """You output ONLY valid JSON (no markdown). Schema:
{"nodes":[{"id":"string","label":"string","kind":"string","scope":"semantic|episodic"}],
 "edges":[{"src":"id","dst":"id","relation":"string"}]}

Node discipline:
- id: stable snake_case identifier; reuse the same id when the same entity appears again.
- kind: fact | hypothesis | action | outcome | observation | unparsed
  Use observation for something stated only in this turn; use fact/hypothesis for claims meant to persist.
- scope: semantic (durable beyond this turn) or episodic (this-turn utterance, step, or local detail).

Relations MUST be exactly one of these (Pearl-style: text cannot prove causation):
- association: co-occurrence or correlation, no directional claim
- hypothesized_influence: guessed direction of influence (NOT verified intervention)
- temporal_succession: A before B in time or narrative order only
- evidence_supports: one statement supports another as evidence
- evidence_conflicts: statements conflict

Never use the words cause, causal, or effect as relation values.
Hard limits: at most 20 nodes and 40 edges; labels at most 120 characters."""


def extract_graph_delta(
    *,
    task: str,
    assistant_reply: str,
    model: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    **litellm_kwargs: Any,
) -> GraphDelta:
    """
    Call LiteLLM once to parse task + reply into a :class:`GraphDelta`.

    On parse or API failure, returns a single ``unparsed`` node (never raises).

    Args:
        task: User task (truncated by caller if needed).
        assistant_reply: Assistant output text.
        model: LiteLLM model name.
        max_tokens: Cap for extract completion.
        temperature: Low temperature recommended.
        **litellm_kwargs: Passed to ``litellm.completion`` (e.g. api_key, base_url).

    Returns:
        Validated :class:`GraphDelta`.
    """
    user_msg = (
        "Extract a small epistemic graph (associations, time order, hypotheses only) "
        "from this turn. Do not assert true causal mechanisms.\n\n"
        f"TASK:\n{task}\n\nASSISTANT:\n{assistant_reply}\n"
    )
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **litellm_kwargs,
        )
        choice = resp.choices[0].message
        raw = getattr(choice, "content", None) or ""
        if not isinstance(raw, str):
            raw = str(raw)
        raw = _strip_json_fence(raw)
        payload = json.loads(raw)
        delta = GraphDelta.model_validate(payload)
        delta = sanitize_pearl_epistemics(delta)
        logger.debug(
            "world_model extract ok model={} nodes={} edges={}",
            model,
            len(delta.nodes),
            len(delta.edges),
        )
        return delta
    except json.JSONDecodeError as e:
        logger.warning(
            "world_model JSON decode failed: {} (model={})",
            e,
            model,
        )
        return _fallback_delta("json_decode_error")
    except Exception as e:
        logger.warning(
            "world_model extract failed: {} (model={})",
            e,
            model,
        )
        return _fallback_delta(f"extract_error: {type(e).__name__}")


def attach_shared_swarm_wm(
    agents: Sequence[Any],
    shared: SwarmWorldModel,
) -> None:
    """
    Set ``swarm_wm`` on agents that have ``wm=True`` and no swarm graph yet.

    Args:
        agents: Iterable of objects (typically :class:`swarms.structs.agent.Agent`).
        shared: Shared :class:`SwarmWorldModel` instance.
    """
    for a in agents:
        if getattr(a, "wm", False) and getattr(a, "swarm_wm", None) is None:
            a.swarm_wm = shared
