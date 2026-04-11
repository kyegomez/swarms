from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from swarms import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.utils import SwarmWorldModel
from swarms.utils.world_model import (
    AgentWorldModel,
    LinearLatentDynamics,
    attach_shared_swarm_wm,
)

try:
    from voice_agents import StreamingTTSCallback
except ImportError:
    StreamingTTSCallback = None  # type: ignore[misc, assignment]

_EXAMPLE_MODEL = os.environ.get("SWARMS_EXAMPLE_MODEL", "gpt-4o-mini")


def _resolve_wm_throttle_seconds(cli_value: Optional[float]) -> float:
    """
    Seconds to sleep before each LLM/tool step.

    CLI ``--wm-throttle-seconds`` wins when provided; else ``WM_THROTTLE_SECONDS``;
    else ``0.5``. Use ``0`` to disable pacing.
    """
    if cli_value is not None:
        return max(0.0, float(cli_value))
    raw = os.environ.get("WM_THROTTLE_SECONDS")
    if raw is not None and str(raw).strip() != "":
        return max(0.0, float(raw))
    return 0.5


def _apply_wm_loop_throttle_to_agent(agent: Agent, seconds: float) -> None:
    """
    Example-only: sleep before each ``call_llm`` and ``execute_tools`` on this instance.

    This keeps ``swarms/structs/agent.py`` free of demo pacing.
    To remove throttling entirely, delete this function and every call to it below.
    """
    delay = float(seconds or 0.0)
    if delay <= 0:
        return
    _orig_call_llm = agent.call_llm
    _orig_execute_tools = agent.execute_tools

    def call_llm_throttled(*args: Any, **kwargs: Any) -> Any:
        time.sleep(delay)
        return _orig_call_llm(*args, **kwargs)

    def execute_tools_throttled(*args: Any, **kwargs: Any) -> Any:
        time.sleep(delay)
        return _orig_execute_tools(*args, **kwargs)

    agent.call_llm = call_llm_throttled  # type: ignore[method-assign]
    agent.execute_tools = execute_tools_throttled  # type: ignore[method-assign]


AGI_TEAM_TASK = (
    "Shared lab mission: produce a minimal, actionable AGI R&D agenda (capabilities + alignment safety).\n\n"
    "Research Lead: You go first. You MUST start by calling create_plan with at least 3 subtasks "
    "(e.g. measurement/benchmarks, scalable alignment, governance). Use think at least once. "
    "Use filesystem tools: list_directory and/or create_file (e.g. write a short agi_lab_outline.md).\n"
    "Systems Architect: Refine milestones and evaluation; use read_file/create_file/list_directory.\n"
    "Safety Reviewer: Critique risks; use tools — not prose-only.\n\n"
    "Everyone: use Swarms tools (create_plan, think, subtask_done, complete_task, file tools, run_bash "
    "when appropriate). Finish with complete_task."
)

AGI_SOLO_TASK = (
    "You are an AGI research lab assistant. Use the tool loop.\n\n"
    "1. Call create_plan with at least 3 subtasks.\n"
    "2. Use think at least once before completing work.\n"
    "3. Use list_directory, read_file, or create_file (e.g. one paragraph in agi_rd_note.md).\n"
    "4. Use subtask_done and complete_task. Do not answer with prose-only."
)

SP_RESEARCH_LEAD = (
    "You are the Research Lead of an AGI R&D lab. You must use autonomous tools: "
    "create_plan, think, filesystem tools, subtask_done, complete_task."
)

SP_SYSTEMS_ARCHITECT = (
    "You are the Systems Architect. Turn research into testable milestones; use tools "
    "(read_file, create_file, list_directory, think)."
)

SP_SAFETY_REVIEWER = (
    "You are the Safety Reviewer. Stress-test proposals for misuse and governance gaps; use tools."
)

OBS_DIM = 8
ACTION_DIM = 1
LATENT_DIM = 3


def graph_observation_vector(wm: AgentWorldModel | SwarmWorldModel) -> np.ndarray:
    """Map current graph ``to_dict()`` to a numeric observation (length ``OBS_DIM``)."""
    d = wm.to_dict()
    nodes: List[Dict[str, Any]] = d["nodes"]
    edges: List[Dict[str, Any]] = d["edges"]
    nn, ne = len(nodes), len(edges)
    facts = sum(1 for n in nodes if n.get("kind") == "fact")
    hyp = sum(1 for n in nodes if n.get("kind") == "hypothesis")
    obs_k = sum(1 for n in nodes if n.get("kind") == "observation")
    sem = sum(1 for n in nodes if n.get("scope") == "semantic")
    epi = sum(1 for n in nodes if n.get("scope") == "episodic")
    density = float(ne / max(nn, 1))
    v = [
        float(nn),
        float(ne),
        float(facts),
        float(hyp),
        float(obs_k),
        float(sem),
        float(epi),
        density,
    ]
    if len(v) != OBS_DIM:
        raise RuntimeError("OBS_DIM must match graph_observation_vector length")
    return np.array(v, dtype=np.float64)


def make_trajectory_callback_for_wm(
    wm: AgentWorldModel | SwarmWorldModel,
    *,
    action_dim: int = ACTION_DIM,
) -> Callable[..., None]:
    """
    After each ``on_wm_update``, record (prev_obs, ones, obs) on ``wm.trajectory_dynamics``.

    Call ``wm.init_trajectory_dynamics(...)`` before the agent runs.
    """
    prev: Dict[str, Optional[np.ndarray]] = {"v": None}
    action = np.ones(action_dim, dtype=np.float64)

    def on_wm_update(
        _wm: AgentWorldModel,
        digest: str,
        *,
        loop_count: Optional[int],
        task_excerpt: str,
    ) -> None:
        td = wm.trajectory_dynamics
        if td is None:
            return
        obs = graph_observation_vector(wm)
        if prev["v"] is not None:
            td.observe(prev["v"], action, obs)
        prev["v"] = obs.copy()

    return on_wm_update


def _trajectory_summary(td: Optional[LinearLatentDynamics], label: str) -> Dict[str, Any]:
    """Serialize trajectory dynamics state for logging."""
    if td is None:
        return {"label": label, "enabled": False}
    d = td.to_dict()
    d["label"] = label
    d["enabled"] = True
    d["is_fitted"] = td.is_fitted
    return d


def _require_llm_env() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Set OPENAI_API_KEY (or your provider env vars) so the agent and "
            "world-model extract call can run.",
        )
        return False
    return True


def on_wm_update(wm: Any, digest: str, *, loop_count: Any, task_excerpt: str) -> None:
    """Default world-model update callback: log to stdout."""
    print(f"[wm] loop={loop_count} digest={digest!r}")


def main(
    *,
    viz: bool,
    quick: bool,
    wm_throttle_seconds: float,
) -> None:
    """Run a single agent with world-model integration."""
    if not _require_llm_env():
        return

    if quick:
        max_loops: Any = 5 if viz else 1
        run_task = (
            "In three short bullets: one benefit of exercise, one of hydration, one of sleep."
        )
    else:
        max_loops = "auto"
        run_task = AGI_SOLO_TASK

    agent = Agent(
        agent_name="AGI-Lab-Solo",
        model_name=_EXAMPLE_MODEL,
        max_loops=max_loops,
        selected_tools="all",
        interactive=False,
        system_prompt=SP_RESEARCH_LEAD,
        wm=True,
        wm_max_chars=4000,
        on_wm_update=on_wm_update,
        print_on=True,
        verbose=True,
    )
    _apply_wm_loop_throttle_to_agent(agent, wm_throttle_seconds)

    g = agent.world_model_graph
    if viz and g is not None:
        g.init_trajectory_dynamics(OBS_DIM, ACTION_DIM, LATENT_DIM, seed=42)
        _track = make_trajectory_callback_for_wm(g)

        def _wm_cb(wm: Any, digest: str, *, loop_count: Any, task_excerpt: str) -> None:
            print(f"[wm] loop={loop_count} digest={digest!r}")
            _track(wm, digest, loop_count=loop_count, task_excerpt=task_excerpt)

        agent.on_wm_update = _wm_cb

    out = agent.run(run_task)
    print(out)

    if g is not None:
        print("--- world model (agent) ---")
        print("digest:", g.digest_for_hook())
        d = g.to_dict()
        print(f"nodes: {len(d['nodes'])}, edges: {len(d['edges'])}")
        if viz and g.trajectory_dynamics is not None:
            ok = g.trajectory_dynamics.fit()
            print(
                f"trajectory_dynamics: n_samples={g.trajectory_dynamics.n_samples} "
                f"fitted={ok}",
            )


def demo_rearrange_wm(
    *,
    voice: bool,
    viz: bool,
    wm_throttle_seconds: float,
) -> None:
    """Run a three-agent team with shared world model."""
    if not _require_llm_env():
        return

    if voice and StreamingTTSCallback is None:
        print(
            "voice-agents is not installed. Install with: pip install voice-agents\n"
            "Or run: python examples/utils/wm_example.py --rearrange --no-voice",
        )
        return

    shared = SwarmWorldModel("agi-rd-shared-wm")

    if viz:
        shared.init_trajectory_dynamics(OBS_DIM, ACTION_DIM, LATENT_DIM, seed=42)
        _track = make_trajectory_callback_for_wm(shared)

        def wm_cb(wm: Any, digest: str, *, loop_count: Any, task_excerpt: str) -> None:
            print(f"[wm] loop={loop_count} digest={digest!r}")
            _track(wm, digest, loop_count=loop_count, task_excerpt=task_excerpt)

    else:
        wm_cb = on_wm_update

    tts_lead = (
        StreamingTTSCallback(voice="onyx", model="openai/tts-1")
        if voice
        else None
    )
    tts_arch = (
        StreamingTTSCallback(voice="nova", model="openai/tts-1")
        if voice
        else None
    )
    tts_safe = (
        StreamingTTSCallback(voice="shimmer", model="openai/tts-1")
        if voice
        else None
    )

    research_lead = Agent(
        agent_name="Research Lead",
        model_name=_EXAMPLE_MODEL,
        max_loops="auto",
        selected_tools="all",
        interactive=False,
        system_prompt=SP_RESEARCH_LEAD,
        wm=True,
        wm_max_chars=4000,
        on_wm_update=wm_cb,
        streaming_on=voice,
        print_on=not voice,
        verbose=True,
        streaming_callback=tts_lead,
    )
    systems_architect = Agent(
        agent_name="Systems Architect",
        model_name=_EXAMPLE_MODEL,
        max_loops="auto",
        selected_tools="all",
        interactive=False,
        system_prompt=SP_SYSTEMS_ARCHITECT,
        wm=True,
        wm_max_chars=4000,
        on_wm_update=wm_cb,
        streaming_on=voice,
        print_on=not voice,
        verbose=True,
        streaming_callback=tts_arch,
    )
    safety_reviewer = Agent(
        agent_name="Safety Reviewer",
        model_name=_EXAMPLE_MODEL,
        max_loops="auto",
        selected_tools="all",
        interactive=False,
        system_prompt=SP_SAFETY_REVIEWER,
        wm=True,
        wm_max_chars=4000,
        on_wm_update=wm_cb,
        streaming_on=voice,
        print_on=not voice,
        verbose=True,
        streaming_callback=tts_safe,
    )

    for _a in (research_lead, systems_architect, safety_reviewer):
        _apply_wm_loop_throttle_to_agent(_a, wm_throttle_seconds)

    attach_shared_swarm_wm(
        [research_lead, systems_architect, safety_reviewer],
        shared,
    )

    swarm = AgentRearrange(
        name="AGI-RD-Lab",
        agents=[research_lead, systems_architect, safety_reviewer],
        flow="Research Lead -> Systems Architect -> Safety Reviewer",
        max_loops=1,
        verbose=False,
        team_awareness=False,
        output_type="all",
        autosave=False,
    )

    out = swarm.run(AGI_TEAM_TASK)
    print(out)

    if voice:
        if tts_lead:
            tts_lead.flush()
        if tts_arch:
            tts_arch.flush()
        if tts_safe:
            tts_safe.flush()

    print("--- shared swarm graph ---")
    print("digest:", shared.digest_for_hook())
    sd = shared.to_dict()
    print(f"nodes: {len(sd['nodes'])}, edges: {len(sd['edges'])}")

    if viz and shared.trajectory_dynamics is not None:
        ok = shared.trajectory_dynamics.fit()
        print(
            f"trajectory_dynamics (shared): n_samples={shared.trajectory_dynamics.n_samples} "
            f"fitted={ok}",
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AGI R&D lab demo (autonomous tools + world-model).",
    )
    p.add_argument(
        "--rearrange",
        action="store_true",
        help=(
            "Run AGI R&D team (Research Lead -> Systems Architect -> Safety Reviewer) "
            "with max_loops='auto' and selected_tools='all'."
        ),
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Solo mode only: short integer max_loops and a tiny task (no max_loops='auto').",
    )
    p.add_argument(
        "--shared",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable TTS (no voice-agents); rearrange mode only.",
    )
    p.add_argument(
        "--viz",
        action="store_true",
        help="Enable LinearLatentDynamics trajectory recording and fitting after run.",
    )
    p.add_argument(
        "--wm-throttle-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Pace agent LLM/tool steps (min seconds between calls). "
            "Default: WM_THROTTLE_SECONDS or 0.5. Use 0 to disable."
        ),
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    wm_throttle_seconds = _resolve_wm_throttle_seconds(args.wm_throttle_seconds)
    use_rearrange = args.rearrange or args.shared
    if use_rearrange:
        demo_rearrange_wm(
            voice=not args.no_voice,
            viz=args.viz,
            wm_throttle_seconds=wm_throttle_seconds,
        )
    else:
        main(
            viz=args.viz,
            quick=args.quick,
            wm_throttle_seconds=wm_throttle_seconds,
        )
