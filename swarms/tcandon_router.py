"""Simple, interpretable implementation of the TCAndon-Router idea.

This is a lightweight, dependency-free router suitable for local use and
unit-testing. It accepts a list of agents (dicts with at least
`id`, `name`, and `description`) and a query string, then returns a
ranked selection with short, transparent reasons.

This is not a reproduction of the original training pipeline (SFT+DAPO)
or the model weights, but a practical, easily-extended router inspired
by TCAndon-Router's goals: transparent routing, multi-agent preservation,
and explainable decisions.

Usage example:
    from swarms.tcandon_router import TCAndonRouter

    agents = [
        {"id": "weather", "name": "WeatherAgent", "description": "Provides weather forecasts and alerts."},
        {"id": "maps", "name": "MapsAgent", "description": "Helps with directions and points of interest."},
    ]

    router = TCAndonRouter(run="default", max_agents=2)
    result = router.run(agents, "Is there a pub near MG Road tonight?")
    print(result)

"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


class TCAndonRouter:
    """Lightweight TCAndon-style router.

    Args:
        run: mode name (free-form). Kept for API parity with requested
            "run parameter"; can be used by callers to indicate different
            routing behaviors (e.g. "explain", "fast", "llm").
        max_agents: maximum number of agents to return when multiple
            agents are applicable.
        oos_threshold: minimum score required to consider an agent
            relevant. If no agent meets this threshold the router
            returns an "oos" (out-of-scope) style empty selection.
    """

    def __init__(self, run: str = "default", max_agents: int = 3, oos_threshold: float = 0.08):
        # store `run` as `mode` to avoid shadowing the `run()` method
        self.mode = run
        self.max_agents = max_agents
        self.oos_threshold = float(oos_threshold)

    def _score_agent(self, query: str, agent: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score an agent for the given query. Returns (score, meta).

        The score is a simple combination of token overlap and string
        similarity to keep the implementation dependency-free and
        interpretable. The returned meta contains tokens used for
        generating a short explanation.
        """
        q_tokens = set(_tokenize(query))
        desc = (agent.get("description") or agent.get("desc") or "")
        d_tokens = set(_tokenize(desc))

        # token overlap fraction (how much of the agent description is
        # covered by the query)
        overlap = 0.0
        if d_tokens:
            overlap = len(q_tokens & d_tokens) / float(len(d_tokens))

        # rough surface similarity (captures phrase-level similarity)
        sim = SequenceMatcher(None, query.lower(), desc.lower()).ratio()

        # weighted score (these weights are intentionally simple and
        # easy to tweak)
        score = 0.65 * overlap + 0.35 * sim

        meta = {
            "overlap": overlap,
            "similarity": sim,
            "matched_tokens": sorted(list(q_tokens & d_tokens)),
        }
        return float(score), meta

    def run(self, agents: List[Dict[str, Any]], query: str, top_k: int | None = None) -> Dict[str, Any]:
        """Route a query to the most relevant agents.

        Args:
            agents: list of agent descriptions. Each agent should be a
                dict containing `id`, `name` (optional) and `description`.
            query: the user's query string.
            top_k: override for number of agents to return.

        Returns a dictionary containing:
            - "selected": list of selected agent ids (may be empty for oos)
            - "reasons": human-friendly reason string
            - "scores": list of dicts with per-agent scores and meta
        """
        if not isinstance(agents, list):
            raise TypeError("agents must be a list of dicts")

        top_k = top_k or self.max_agents

        scored = []
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            score, meta = self._score_agent(query or "", agent)
            scored.append({"agent": agent, "score": score, "meta": meta})

        scored.sort(key=lambda x: x["score"], reverse=True)

        if not scored:
            return {"selected": [], "reasons": "no agents provided", "scores": []}

        top_score = scored[0]["score"]
        if top_score < self.oos_threshold:
            # out-of-scope
            return {
                "selected": [],
                "reasons": f"oos: top score {top_score:.3f} below threshold {self.oos_threshold}",
                "scores": scored,
            }

        chosen = scored[:top_k]
        selected_ids = [c["agent"].get("id") or c["agent"].get("ID") or c["agent"].get("name") for c in chosen]

        # Build compact reason: list top agents and highlight matched tokens
        parts = []
        for c in chosen:
            a = c["agent"]
            aid = a.get("id") or a.get("ID") or a.get("name")
            score = c["score"]
            tokens = c["meta"].get("matched_tokens") or []
            if tokens:
                parts.append(f"{aid} (score={score:.3f}, matched={','.join(tokens)})")
            else:
                parts.append(f"{aid} (score={score:.3f})")

        reason = "Selected agents: " + "; ".join(parts)

        return {"selected": selected_ids, "reasons": reason, "scores": scored}


__all__ = ["TCAndonRouter"]
