"""Evaluator utilities for auto-swarm builder.

Provides dataset loading, evaluation, and a simple judge/critic to
recommend and rebuild agents between iterations.
"""
from typing import List, Dict, Any, Optional
import random
import statistics


def load_eval_dataset(path: str) -> Optional[List[Dict[str, Any]]]:
    try:
        import json
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return None


def evaluate_agents(agents: List[str], dataset: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate each agent on the dataset and return normalized scores.

    This function mirrors the lightweight simulation used by the example
    UI. If `config['use_real_agents']` is True and a `swarms.structs.agent.Agent`
    class exists, it will attempt to use it; otherwise evaluation is
    deterministically pseudo-random per-agent for reproducible demos.
    """
    scores: Dict[str, float] = {}
    use_real = config.get("use_real_agents", False)
    real_agent_class = None
    if use_real:
        try:
            from swarms.structs.agent import Agent as RealAgent
            real_agent_class = RealAgent
        except Exception:
            real_agent_class = None

    for a in agents:
        correct = 0
        if real_agent_class:
            try:
                agent = real_agent_class(agent_name=a, model_name=config.get("model"))
                for case in dataset:
                    out = agent.run(task=case["input"]) or ""
                    if str(case.get("expected", "")).lower() in str(out).lower():
                        correct += 1
            except Exception:
                # fallback to simulated scoring
                seed = abs(hash(a)) % (2 ** 32)
                rng = random.Random(seed)
                for case in dataset:
                    resp = rng.choice(case["options"])
                    if resp == case["expected"]:
                        correct += 1
        else:
            seed = abs(hash(a)) % (2 ** 32)
            rng = random.Random(seed)
            for case in dataset:
                resp = rng.choice(case["options"])
                if resp == case["expected"]:
                    correct += 1
        scores[a] = correct / max(1, len(dataset))
    return scores


def judge_and_improve(agents: List[str], scores: Dict[str, float], config: Dict[str, Any], iteration: int = 1) -> List[str]:
    """Simple judge that keeps top performers and rebuilds others.

    Rebuilt agents are created with a versioned name and a short critic
    suggestion encoded into the name to indicate intended improvement.
    """
    if not agents:
        return agents
    vals = list(scores.values())
    threshold = statistics.median(vals)
    keep = [a for a in agents if scores.get(a, 0) >= threshold]
    remove = [a for a in agents if a not in keep]

    new_agents = keep.copy()
    rebuilt_count = 0
    for r in remove:
        rebuilt_count += 1
        # critic: suggest a role tweak based on failure pattern (simple heuristic)
        suggestion = "focus-signal" if scores.get(r, 0) < 0.5 else "refine"
        new_agents.append(f"{r}_rebuilt_{iteration}_{rebuilt_count}_{suggestion}")

    # ensure list length matches original
    while len(new_agents) < len(agents):
        new_agents.append(f"agent_extra_{iteration}_{len(new_agents)}")

    return new_agents
