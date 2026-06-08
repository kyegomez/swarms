"""Agent factories for ``HeavySwarm``.

This module owns every Agent specification used by ``HeavySwarm`` — the
default 5-agent layout, the 4-agent Grok variant, and the 16-agent Grok
Heavy variant — plus a single ``build_heavy_swarm_agents`` dispatcher.
Keeps the orchestration class in ``swarms/structs/heavy_swarm.py`` free
of prompt imports and per-role boilerplate.
"""

from typing import Dict, List, Literal, Optional, Tuple


from swarms.prompts.heavy_swarm_prompts import (
    ALTERNATIVES_AGENT_PROMPT,
    ANALYSIS_AGENT_PROMPT,
    BENJAMIN_HEAVY_PROMPT,
    BENJAMIN_PROMPT,
    CAPTAIN_SWARM_PROMPT,
    CHARLOTTE_PROMPT,
    ELIZABETH_PROMPT,
    GROK_HEAVY_CAPTAIN_PROMPT,
    HARPER_HEAVY_PROMPT,
    HARPER_PROMPT,
    HENRY_PROMPT,
    JACK_PROMPT,
    JAMES_PROMPT,
    LUCAS_HEAVY_PROMPT,
    LUCAS_PROMPT,
    LUNA_PROMPT,
    MIA_PROMPT,
    NOAH_PROMPT,
    OLIVIA_PROMPT,
    OWEN_PROMPT,
    RESEARCH_AGENT_PROMPT,
    SEBASTIAN_PROMPT,
    SYNTHESIS_AGENT_PROMPT,
    VERIFICATION_AGENT_PROMPT,
    WILLIAM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.tools.tool_type import tool_type


# (key, agent_name, agent_description, system_prompt, is_leader)
AgentSpec = Tuple[str, str, str, str, bool]


# All swarm variants
SwarmVariant = Literal["default", "medium", "heavy"]


DEFAULT_SPECS: List[AgentSpec] = [
    (
        "research",
        "Research-Agent",
        "Expert research agent specializing in comprehensive information gathering and data collection",
        RESEARCH_AGENT_PROMPT,
        False,
    ),
    (
        "analysis",
        "Analysis-Agent",
        "Expert analytical agent specializing in pattern recognition, data analysis, and insight generation",
        ANALYSIS_AGENT_PROMPT,
        False,
    ),
    (
        "alternatives",
        "Alternatives-Agent",
        "Expert strategic agent specializing in alternative approaches, creative solutions, and option generation",
        ALTERNATIVES_AGENT_PROMPT,
        False,
    ),
    (
        "verification",
        "Verification-Agent",
        "Expert verification agent specializing in validation, feasibility assessment, and quality assurance",
        VERIFICATION_AGENT_PROMPT,
        False,
    ),
    (
        "synthesis",
        "Synthesis-Agent",
        "Expert synthesis agent specializing in integration, comprehensive analysis, and final recommendations",
        SYNTHESIS_AGENT_PROMPT,
        True,
    ),
]


GROK_SPECS: List[AgentSpec] = [
    (
        "captain",
        "Captain-Swarm",
        "Leader and orchestrator of the Grok Heavy multi-agent system",
        CAPTAIN_SWARM_PROMPT,
        True,
    ),
    (
        "harper",
        "Harper",
        "Research and Facts specialist for evidence-based data gathering and fact verification",
        HARPER_PROMPT,
        False,
    ),
    (
        "benjamin",
        "Benjamin",
        "Logic, Math, and Code specialist for rigorous reasoning and computational verification",
        BENJAMIN_PROMPT,
        False,
    ),
    (
        "lucas",
        "Lucas",
        "Creative and Divergent Thinking specialist for contrarian analysis and blind-spot detection",
        LUCAS_PROMPT,
        False,
    ),
]


GROK_HEAVY_SPECS: List[AgentSpec] = [
    (
        "harper",
        "Harper",
        "Creative Writing and Storytelling specialist",
        HARPER_HEAVY_PROMPT,
        False,
    ),
    (
        "benjamin",
        "Benjamin",
        "Data, Finance and Economics specialist",
        BENJAMIN_HEAVY_PROMPT,
        False,
    ),
    (
        "lucas",
        "Lucas",
        "Coding, Programming and Technical Builds specialist",
        LUCAS_HEAVY_PROMPT,
        False,
    ),
    (
        "olivia",
        "Olivia",
        "Literature, Arts and Culture specialist",
        OLIVIA_PROMPT,
        False,
    ),
    (
        "james",
        "James",
        "History, Politics and Philosophy specialist",
        JAMES_PROMPT,
        False,
    ),
    (
        "charlotte",
        "Charlotte",
        "Math, Statistics and Logic specialist",
        CHARLOTTE_PROMPT,
        False,
    ),
    (
        "henry",
        "Henry",
        "Engineering, Robotics and Innovation specialist",
        HENRY_PROMPT,
        False,
    ),
    (
        "mia",
        "Mia",
        "Biology, Health and Medicine specialist",
        MIA_PROMPT,
        False,
    ),
    (
        "william",
        "William",
        "Business Strategy and Entrepreneurship specialist",
        WILLIAM_PROMPT,
        False,
    ),
    (
        "sebastian",
        "Sebastian",
        "Physics, Astronomy and Hard Sciences specialist",
        SEBASTIAN_PROMPT,
        False,
    ),
    (
        "jack",
        "Jack",
        "Psychology and Human Behavior specialist",
        JACK_PROMPT,
        False,
    ),
    (
        "owen",
        "Owen",
        "Environment, Sustainability and Global Systems specialist",
        OWEN_PROMPT,
        False,
    ),
    (
        "luna",
        "Luna",
        "Space Exploration and Futurism specialist",
        LUNA_PROMPT,
        False,
    ),
    (
        "elizabeth",
        "Elizabeth",
        "Ethics, Policy and Critical Thinking specialist",
        ELIZABETH_PROMPT,
        False,
    ),
    (
        "noah",
        "Noah",
        "Long-Term Innovation and Systems Thinking specialist",
        NOAH_PROMPT,
        False,
    ),
    (
        "captain",
        "Grok",
        "Lead coordinator and synthesizer of the 16-agent Grok Heavy system",
        GROK_HEAVY_CAPTAIN_PROMPT,
        True,
    ),
]


def _build_agent(
    *,
    name: str,
    description: str,
    system_prompt: str,
    model_name: str,
    tools: Optional[tool_type] = None,
    is_leader: bool = False,
    worker_prints_on: bool = False,
) -> Agent:
    """Construct a HeavySwarm worker or leader agent.

    Leaders (``is_leader=True``) run a single deterministic loop and always
    print, since they emit the user-facing answer. Workers run with
    ``max_loops="auto"`` and only print when ``worker_prints_on`` is set.
    """
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt,
        model_name=model_name,
        max_loops=1 if is_leader else "auto",
        print_on=True if is_leader else worker_prints_on,
        verbose=False,
        tools=tools,
    )


_SPECS_BY_VARIANT: Dict[str, List[AgentSpec]] = {
    "default": DEFAULT_SPECS,
    "medium": GROK_SPECS,
    "heavy": GROK_HEAVY_SPECS,
}


def build_heavy_swarm_agents(
    *,
    model_name: str,
    tools: Optional[tool_type] = None,
    worker_prints_on: bool = False,
    variant: SwarmVariant = "default",
) -> Dict[str, Agent]:
    """Build the dict of named agents for a HeavySwarm variant.

    ``variant`` picks the spec table:
    ``"default"`` (5 agents), ``"medium"`` (4 agents), ``"heavy"`` (16 agents).
    """
    try:
        specs = _SPECS_BY_VARIANT[variant]
    except KeyError:
        raise ValueError(
            f"Unknown variant {variant!r}; "
            f"expected one of {list(_SPECS_BY_VARIANT)}"
        )

    return {
        key: _build_agent(
            name=name,
            description=desc,
            system_prompt=prompt,
            model_name=model_name,
            tools=tools,
            is_leader=is_leader,
            worker_prints_on=worker_prints_on,
        )
        for key, name, desc, prompt, is_leader in specs
    }
