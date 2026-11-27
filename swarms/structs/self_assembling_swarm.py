"""
SelfAssemblingSwarm: agents form their own teams and search for the
architecture that solves a task best.

This implements Coalition Architecture Search. The agents split the task
into subgoals, form teams through stable matching, choose how each team
works inside, link teams with channels, and reorganise from measured
credit until the structure stops improving. The best structure is kept
as a harness that can run again without the search.

Every message an agent sends is a structured tool call. At each step
the agent is given a set of message tools, must call one of them, and
chooses which: for example submit_output or request_input while
working, approve_draft or request_changes while reviewing, and
raise_challenge or endorse_output when it reads another team's work.
"""

import json
import math
import random
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from swarms.structs.agent import Agent
from swarms.structs.context_utils import agent_answer
from swarms.structs.conversation import Conversation
from swarms.structs.usage_utils import total_usage
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)
from swarms.tools.tool_choice import forced_tool_choice
from swarms.utils.generate_id import generate_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.utils.workspace_manager import WorkspaceManager

logger = initialize_logger(log_folder="self_assembling_swarm")

INTEGRATE = "integrate"

PROTOCOL_MIN_SIZE: Dict[str, int] = {
    "SOLO": 1,
    "CHAIN": 2,
    "FANOUT": 2,
    "CRITIC_LOOP": 2,
    "LEAD": 2,
    "DEBATE": 3,
    "VOTE": 3,
}

PROTOCOLS: Tuple[str, ...] = tuple(PROTOCOL_MIN_SIZE)

NO_OUTPUT = "(This team produced no output.)"

_MISSING = object()


def _tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: List[str],
) -> Dict[str, Any]:
    """Build one message tool in the OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _text(description: str) -> Dict[str, Any]:
    """Return a string field schema."""
    return {"type": "string", "description": description}


def _number(description: str, high: float = 1.0) -> Dict[str, Any]:
    """Return a bounded number field schema."""
    return {
        "type": "number",
        "minimum": 0,
        "maximum": high,
        "description": description,
    }


def _choice(values: List[str], description: str) -> Dict[str, Any]:
    """Return a string field limited to some values."""
    return {
        "type": "string",
        "enum": values,
        "description": description,
    }


def _strings(description: str) -> Dict[str, Any]:
    """Return a list-of-strings field schema."""
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": description,
    }


PROPOSE_SUBGOALS = _tool(
    "propose_subgoals",
    "Propose how to split the task into subgoals that separate teams can work on.",
    {
        "subgoals": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "id": _text("A short id such as s1."),
                    "description": _text("What the subgoal is."),
                    "kind": _text(
                        "A one or two word category, such as research, code, analysis, writing or review."
                    ),
                    "depends_on": _strings(
                        "Ids of the subgoals whose output this one needs."
                    ),
                    "output": _text("What the subgoal produces."),
                },
                "required": [
                    "id",
                    "description",
                    "kind",
                    "depends_on",
                    "output",
                ],
            },
        },
        "reason": _text("Why this split fits the task."),
    },
    ["subgoals", "reason"],
)

KEEP_WHOLE_TASK = _tool(
    "keep_whole_task",
    "Say the task should not be split, because one team should do all of it.",
    {"reason": _text("Why the task should stay whole.")},
    ["reason"],
)

SUBMIT_OUTPUT = _tool(
    "submit_output",
    "Deliver your output for this step to your team and to the teams that use it.",
    {
        "output": _text("The full output."),
        "summary": _text(
            "One or two sentences on what the output contains."
        ),
        "confidence": _number(
            "How sure you are that the output is right, from 0 to 1."
        ),
        "assumptions": _strings(
            "Anything you assumed because an input was missing or unclear."
        ),
    },
    ["output", "summary", "confidence", "assumptions"],
)

ANSWER_REQUEST = _tool(
    "answer_request",
    "Answer another team's question.",
    {"answer": _text("Your answer.")},
    ["answer"],
)

DECLINE_REQUEST = _tool(
    "decline_request",
    "Decline a question your team cannot answer.",
    {"reason": _text("Why your team cannot help.")},
    ["reason"],
)

APPROVE_DRAFT = _tool(
    "approve_draft",
    "Approve a teammate's draft as ready.",
    {"note": _text("Anything worth keeping in mind.")},
    ["note"],
)

REQUEST_CHANGES = _tool(
    "request_changes",
    "Send a teammate's draft back with the problems to fix.",
    {
        "issues": _strings("Each specific problem to fix."),
        "critique": _text("The review in a few sentences."),
    },
    ["issues", "critique"],
)

RAISE_CHALLENGE = _tool(
    "raise_challenge",
    "Challenge another team's output because of a concrete problem.",
    {
        "claim": _text("The problem."),
        "evidence": _text("What shows it is a problem."),
        "severity": _choice(
            ["low", "medium", "high"],
            "How much it would hurt the answer.",
        ),
    },
    ["claim", "evidence", "severity"],
)

ENDORSE_OUTPUT = _tool(
    "endorse_output",
    "Accept another team's output without a challenge.",
    {"note": _text("Anything worth keeping in mind.")},
    ["note"],
)

_REPLIES = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "team": _text("The team that raised the challenge."),
            "accepted": {
                "type": "boolean",
                "description": "Whether the challenge was right.",
            },
            "reply": _text("Your answer to the challenge."),
        },
        "required": ["team", "accepted", "reply"],
    },
    "description": "One reply per challenge.",
}

REVISE_OUTPUT = _tool(
    "revise_output",
    "Change your team's output to answer the challenges.",
    {
        "output": _text("The full revised output."),
        "replies": _REPLIES,
    },
    ["output", "replies"],
)

STAND_BY_OUTPUT = _tool(
    "stand_by_output",
    "Keep your team's output as it is and reply to each challenge.",
    {"replies": _REPLIES},
    ["replies"],
)

SPLIT_SUBGOAL = _tool(
    "split_subgoal",
    "Split your team's subgoal into two that two teams can do in parallel.",
    {
        "parts": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "description": _text("What the part is."),
                    "output": _text("What the part produces."),
                },
                "required": ["description", "output"],
            },
        }
    },
    ["parts"],
)

KEEP_SUBGOAL = _tool(
    "keep_subgoal",
    "Keep your team's subgoal whole.",
    {"reason": _text("Why splitting would not help.")},
    ["reason"],
)

SCORE_ANSWER = _tool(
    "score_answer",
    "Score how well an answer completes a task.",
    {
        "score": _number(
            "0 for useless or wrong, 10 for complete and correct.", 10
        ),
        "reason": _text("One sentence on the score."),
    },
    ["score", "reason"],
)


WRITE_RUBRIC = _tool(
    "write_rubric",
    "List the criteria a complete, correct answer to a task must meet.",
    {
        "criteria": {
            "type": "array",
            "minItems": 4,
            "maxItems": 10,
            "items": {"type": "string"},
            "description": "Each specific thing a complete, correct answer must do.",
        }
    },
    ["criteria"],
)

COMPARE_ANSWERS = _tool(
    "compare_answers",
    "Say which of two answers completes the task better, and by how much.",
    {
        "better": _choice(
            ["first", "second", "same"],
            "Which answer completes the task better.",
        ),
        "margin": _number(
            "How much better: 0 for no real difference, 1 for decisive."
        ),
        "reason": _text("One sentence on the difference."),
    },
    ["better", "margin", "reason"],
)


def _score_rubric(count: int) -> Dict[str, Any]:
    """Build the tool the judge uses to score an answer on each criterion."""
    return _tool(
        "score_rubric",
        "Score an answer on every criterion of the rubric.",
        {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": count,
                            "description": "The criterion's number.",
                        },
                        "score": _number(
                            "0 not met, 1 partly, 2 mostly, 3 fully.",
                            3,
                        ),
                        "note": _text("One short note."),
                    },
                    "required": ["criterion", "score", "note"],
                },
            }
        },
        ["scores"],
    )


def _consolidate_subgoals(names: List[str]) -> Dict[str, Any]:
    """Build the tool an agent uses to merge everyone's proposed subgoals."""
    return _tool(
        "consolidate_subgoals",
        "Merge the agents' proposals into one list of distinct subgoals.",
        {
            "subgoals": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": _text("A short id such as s1."),
                        "description": _text("What the subgoal is."),
                        "kind": _text("A one or two word category."),
                        "depends_on": _strings(
                            "Ids, from this list, of the subgoals whose output this one needs."
                        ),
                        "output": _text("What the subgoal produces."),
                        "proposed_by": {
                            "type": "array",
                            "items": _choice(names, "An agent."),
                            "description": "Every agent whose proposal included this subgoal or a close variant of it.",
                        },
                    },
                    "required": [
                        "id",
                        "description",
                        "kind",
                        "depends_on",
                        "output",
                        "proposed_by",
                    ],
                },
            }
        },
        ["subgoals"],
    )


def _rate_subgoals(ids: List[str]) -> Dict[str, Any]:
    """Build the tool an agent uses to rate its fit for each subgoal."""
    return _tool(
        "rate_subgoals",
        "Rate how well you would do each subgoal.",
        {
            "ratings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subgoal": _choice(ids, "The subgoal id."),
                        "score": _number(
                            "0 if you would do it poorly, 1 if very well."
                        ),
                        "reason": _text("One short reason."),
                    },
                    "required": ["subgoal", "score", "reason"],
                },
            }
        },
        ["ratings"],
    )


def _request_input(teams: List[str]) -> Dict[str, Any]:
    """Build the tool an agent uses to ask another team a question."""
    return _tool(
        "request_input",
        "Ask another team for something you need before you can finish.",
        {
            "team": _choice(teams, "The team to ask."),
            "question": _text("What you need from it."),
        },
        ["team", "question"],
    )


def _cast_vote(labels: List[str]) -> Dict[str, Any]:
    """Build the tool an agent uses to vote for a candidate."""
    return _tool(
        "cast_vote",
        "Vote for the candidate output that best meets the subgoal.",
        {
            "candidate": _choice(labels, "The candidate's label."),
            "reason": _text("One short reason."),
        },
        ["candidate", "reason"],
    )


def _assign_parts(workers: List[str]) -> Dict[str, Any]:
    """Build the tool a team lead uses to hand out parts of a subgoal."""
    return _tool(
        "assign_parts",
        "Give each teammate one part of the subgoal to produce.",
        {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": _choice(workers, "The teammate."),
                        "part": _text(
                            "The part they should produce."
                        ),
                    },
                    "required": ["agent", "part"],
                },
            }
        },
        ["assignments"],
    )


def _rate_teammates(names: List[str]) -> Dict[str, Any]:
    """Build the tool an agent uses to rate its teammates."""
    return _tool(
        "rate_teammates",
        "Rate how much each teammate contributed to the team's output.",
        {
            "ratings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": _choice(names, "The teammate."),
                        "score": _number(
                            "0 for nothing, 1 for most of the output."
                        ),
                        "reason": _text("One short reason."),
                    },
                    "required": ["agent", "score", "reason"],
                },
            }
        },
        ["ratings"],
    )


DECOMPOSE_PROMPT = """You are one of {count} agents who will organise themselves into teams to complete a task.

Task:
{task}

Propose how to split it into at most {max_subgoals} subgoals that separate teams can work on, some of them in parallel, or say it should stay whole. Leave out the final step that combines everything; it is added for you.{generic}"""

GENERIC_NOTE = "\nWrite each subgoal so it applies to any task of this kind, not only to this one."

ASSESS_PROMPT = """Task:
{task}

The swarm split the task into these subgoals:

{subgoals}

Rate how well you, with your skills and role, would do each subgoal."""

DRAFT = "Produce the subgoal's output and submit it. If you need something from another team first, you may ask it."

IMPROVE = """A teammate submitted this draft:

{draft}

Improve it: fix errors, fill gaps and keep what is right. Submit the full improved output."""

MERGE = """Your teammates submitted these drafts independently:

{drafts}

Merge them into one output that keeps the best of each, and submit it."""

CRITIQUE = """A teammate submitted this draft:

{draft}

Review it against the subgoal and what it must produce. Approve it if it is ready, or request changes with the specific problems."""

REVISE = """Your draft:

{draft}

Your teammates requested these changes:

{critiques}

Revise the draft to fix every point and submit the full revised output."""

REBUT = """Your output:

{own}

Your teammates proposed these instead:

{others}

Revise your output: keep what you still think is right, adopt better points and answer criticisms. Submit the full output."""

JUDGE = """Your teammates proposed these competing outputs:

{positions}

Decide which points are best supported, write the final output from them, and submit it."""

BALLOT = """Your team produced these candidate outputs:

{candidates}

Vote for the candidate that best meets the subgoal."""

PLAN = "You lead this team, with teammates {names}. Assign each teammate a part of the subgoal, or do the subgoal yourself and submit the output."

PART = """Your team lead assigned you this part:

{part}

Produce it and submit it."""

ASSEMBLE = """Your teammates submitted these parts:

{parts}

Assemble them into the subgoal's complete output and submit it."""

CHALLENGE = """Overall task:
{task}

You are on team {own_id}, which owns this subgoal: {own}
Team {other_id}, which owns "{other}", submitted this output:

{artifact}

If it has a concrete problem that would hurt your team's work or the final answer, challenge it with evidence. Otherwise endorse it."""

ANSWER_CHALLENGES = """Your team's current output:

{artifact}

Other teams raised these challenges:

{challenges}

Revise the output where a challenge is right, or stand by it where the challenge is wrong. Reply to every challenge."""

REQUEST_PROMPT = """Overall task:
{task}

You are on team {own_id}, which owns this subgoal: {own}
{current}
Team {asker_id}, which owns "{asker}", asks your team:

{question}

Answer it, or decline if your team cannot help."""

REQUEST_REPLY = """The reply to your request:

{reply}

Now finish your step."""

PEER_REVIEW = """Overall task:
{task}

Your team worked on this subgoal: {subgoal}
What each member submitted:

{trace}

Rate how much each of your teammates contributed to the team's output."""

SPLIT_PROMPT = """Overall task:
{task}

Your team's subgoal is: {subgoal}
It must produce: {output}

The swarm's measurements suggest this subgoal is too big or its team too divided. Split it into two subgoals that two teams can do in parallel, or keep it whole if splitting would not help."""

CONSOLIDATE_PROMPT = """Task:
{task}

Each agent proposed how to split the task:

{proposals}

Merge the proposals into at most {limit} distinct subgoals. Treat subgoals that cover the same work as one, even when worded differently, and keep every distinct piece of work that matters. For each subgoal, list every agent whose proposal included it. Leave out the final step that combines everything; it is added for you."""

RUBRIC_PROMPT = """Task:
{task}

List the criteria a complete, correct answer to this task must meet. Make each one specific enough that an answer with a missing or wrong part fails it."""

RUBRIC_SCORE_PROMPT = """Task:
{task}

Criteria:
{criteria}

Answer:
{answer}

Score the answer on every criterion: 0 not met, 1 partly met, 2 mostly met, 3 fully met."""

COMPARE_PROMPT = """Task:
{task}

First answer:
{first}

Second answer:
{second}

Say which answer completes the task better, and by how much. Judge what each answer covers and gets right, not its length."""

VERIFIER_PROMPT = (
    "You are an independent, strict judge. You score how well an "
    "answer completes a task. Judge only the answer, not how it was "
    "produced."
)

VERIFY_TASK = """Task:
{task}

Answer:
{answer}

Score how well the answer completes the task."""


@dataclass
class CASConfig:
    """Tuning knobs of Coalition Architecture Search.

    The defaults follow SELF_ASSEMBLING_SWARM.md. Each field names the
    symbol it sets in that document.

    Attributes:
        founders_per_subgoal: Agents that may propose a team per subgoal (f).
        self_report_decay: How fast fit moves from self-rating to record (kappa0).
        trust_weight: Weight of past teamwork in the join utility (beta).
        novelty_weight: Weight of the exploration bonus (gamma).
        load_weight: Penalty per team an agent is already on (delta).
        join_threshold: Least utility for joining a second team.
        min_support: Drafts that must propose a subgoal for it to survive.
        duplicate_similarity: Word overlap at which two subgoals merge (sigma_dup).
        move_fraction: Share of agents that may move per epoch (rho).
        temperature: Starting exploration temperature (tau0).
        min_temperature: Lowest temperature (tau_min).
        temperature_decay: Factor applied to the temperature each epoch (xi).
        learning_rate: Rate of every learned estimate (eta).
        credit_gain: Scale that maps credit onto the learned estimates.
        own_weight: Weight of measured contribution against peer review (theta).
        upstream_share: Share of a team's credit passed to its inputs.
        channel_bias: Offset on the learned chance of a channel.
        random_channel_rate: Random channels per epoch at full temperature (omega).
        challenge_budget: Challenges each team may send per epoch (B).
        requests_per_step: Questions an agent may ask other teams per step.
        sampled_channels: Challenged teams ablated per episode for channel credit.
        merge_threshold: Traffic ratio above which two teams merge (theta_m).
        split_threshold: Team connectivity below which a team splits (theta_s).
        low_share: Credit below which a subgoal counts as failing (q_low).
        fail_epochs: Epochs of failure before a split or dissolve (k_f).
        mutation_rate: Chance a team changes protocol at full temperature.
        restart_rate: Chance an epoch restarts from another archived shape.
        critic_rounds: Review rounds in a critic loop team.
        debate_rounds: Rounds in a debate team.
        patience: Epochs without gain before the search stops (w).
        min_improvement: Gain in best fitness that counts as progress (epsilon).
    """

    founders_per_subgoal: int = 2
    self_report_decay: float = 3.0
    trust_weight: float = 0.5
    novelty_weight: float = 0.3
    load_weight: float = 0.2
    join_threshold: float = 0.3
    min_support: int = 2
    duplicate_similarity: float = 0.4
    move_fraction: float = 0.25
    temperature: float = 1.0
    min_temperature: float = 0.05
    temperature_decay: float = 0.7
    learning_rate: float = 0.3
    credit_gain: float = 5.0
    own_weight: float = 0.5
    upstream_share: float = 0.5
    channel_bias: float = -2.0
    random_channel_rate: float = 1.0
    challenge_budget: int = 4
    requests_per_step: int = 1
    sampled_channels: int = 2
    merge_threshold: float = 1.0
    split_threshold: float = 0.25
    low_share: float = 0.05
    fail_epochs: int = 2
    mutation_rate: float = 0.3
    restart_rate: float = 0.1
    critic_rounds: int = 2
    debate_rounds: int = 2
    patience: int = 2
    min_improvement: float = 0.01


@dataclass
class Subgoal:
    """One unit of work in the swarm's plan.

    Attributes:
        id: Unique id within the genome.
        description: What the subgoal is.
        kind: Short category used to share what the swarm learns.
        output: What the subgoal must produce.
        depends_on: Ids of the subgoals whose output this one needs.
        support: How many drafts proposed it.
    """

    id: str
    description: str
    kind: str
    output: str = ""
    depends_on: List[str] = field(default_factory=list)
    support: int = 1


@dataclass
class Team:
    """A coalition of agents that owns one subgoal and runs one protocol.

    Attributes:
        subgoal: Id of the subgoal the team owns.
        members: Agent names, best fit first; the first is the lead.
        protocol: How the team works inside, one of PROTOCOLS.
    """

    subgoal: str
    members: List[str] = field(default_factory=list)
    protocol: str = "SOLO"


@dataclass
class Genome:
    """A whole architecture: subgoals, their teams, and extra channels.

    Dependency channels follow from the subgoal graph. The channels field
    holds only the learned and random links, keyed by a sorted pair of
    subgoal ids.

    Attributes:
        subgoals: Subgoals by id, including the integrate subgoal.
        teams: The team on each subgoal, keyed by subgoal id.
        channels: Extra links between teams, mapped to learned or random.
    """

    subgoals: Dict[str, Subgoal]
    teams: Dict[str, Team] = field(default_factory=dict)
    channels: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def work_ids(self) -> List[str]:
        """Return every subgoal id except the integrate subgoal.

        Returns:
            Subgoal ids in insertion order.
        """
        return [gid for gid in self.subgoals if gid != INTEGRATE]

    def dependents(self, gid: str) -> List[str]:
        """Return the subgoals that need a subgoal's output.

        Args:
            gid: The subgoal id.

        Returns:
            Ids of the subgoals that depend on it.
        """
        return [
            other.id
            for other in self.subgoals.values()
            if gid in other.depends_on
        ]

    def refresh(self) -> None:
        """Drop dead dependencies and point integrate at every leaf."""
        work = self.work_ids()

        for gid in work:
            subgoal = self.subgoals[gid]
            subgoal.depends_on = [
                dep
                for dep in dict.fromkeys(subgoal.depends_on)
                if dep in self.subgoals
                and dep not in (INTEGRATE, gid)
            ]

        needed = {
            dep
            for gid in work
            for dep in self.subgoals[gid].depends_on
        }
        self.subgoals[INTEGRATE].depends_on = [
            gid for gid in work if gid not in needed
        ]

        self.teams = {
            gid: team
            for gid, team in self.teams.items()
            if gid in self.subgoals
        }
        self.channels = {
            pair: kind
            for pair, kind in self.channels.items()
            if pair[0] in self.subgoals and pair[1] in self.subgoals
        }

    def order(self) -> List[str]:
        """Return the work subgoals in dependency order.

        Returns:
            Ids in topological order; shorter than the work list if the
            graph has a cycle.
        """
        remaining = {
            gid: set(self.subgoals[gid].depends_on)
            for gid in self.work_ids()
        }
        ordered: List[str] = []

        while True:
            ready = [
                gid for gid, deps in remaining.items() if not deps
            ]
            if not ready:
                break
            for gid in ready:
                ordered.append(gid)
                del remaining[gid]
            for deps in remaining.values():
                deps.difference_update(ready)

        return ordered

    def has_cycle(self) -> bool:
        """Return whether the subgoal graph has a cycle.

        Returns:
            True when some subgoals can never become ready.
        """
        return len(self.order()) < len(self.work_ids())

    def waves(self) -> List[List[str]]:
        """Group the subgoals into waves that can run concurrently.

        Returns:
            Lists of subgoal ids; the last wave is the integrate subgoal.
        """
        level: Dict[str, int] = {}

        for gid in self.order():
            deps = self.subgoals[gid].depends_on
            level[gid] = 1 + max(
                (level[dep] for dep in deps), default=0
            )

        waves = [
            [gid for gid in level if level[gid] == depth]
            for depth in range(1, max(level.values(), default=0) + 1)
        ]
        return [wave for wave in waves if wave] + [[INTEGRATE]]

    def challengers(self, target: str) -> List[str]:
        """Return the teams that may challenge a team's output.

        Args:
            target: The subgoal id of the team being challenged.

        Returns:
            Consumers of its output plus teams linked to it by a channel.
        """
        linked = [
            pair[1] if pair[0] == target else pair[0]
            for pair in self.channels
            if target in pair
        ]
        return list(dict.fromkeys(self.dependents(target) + linked))

    def signature(self) -> Set[Tuple[str, ...]]:
        """Return the set of facts that defines this structure.

        Returns:
            Membership, protocol, subgoal and channel facts.
        """
        facts: Set[Tuple[str, ...]] = {
            ("g", gid) for gid in self.subgoals
        }

        for gid, team in self.teams.items():
            facts.add(("p", gid, team.protocol))
            facts.update(("m", gid, name) for name in team.members)

        facts.update(("c", *pair) for pair in self.channels)
        return facts


@dataclass
class _Episode:
    """Everything one run of a genome on one task produced."""

    task: str
    epoch: int = 0
    answer: str = ""
    quality: float = 0.0
    tokens: int = 0
    seconds: float = 0.0
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    shares: Dict[str, Dict[str, float]] = field(default_factory=dict)
    traces: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=dict
    )
    messages: List[Dict[str, Any]] = field(default_factory=list)
    traffic: Counter = field(default_factory=Counter)
    sent: Counter = field(default_factory=Counter)
    requests: Counter = field(default_factory=Counter)
    challenged: Dict[str, List[str]] = field(default_factory=dict)
    pre_revision: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    team_credit: Dict[str, float] = field(default_factory=dict)
    agent_credit: Dict[Tuple[str, str], float] = field(
        default_factory=dict
    )
    ratings: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )
    channel_credit: Dict[Tuple[str, str], float] = field(
        default_factory=dict
    )

    def output(self, gid: str) -> str:
        """Return the text a team submitted, or the no-output marker."""
        return self.artifacts.get(gid, {}).get("output") or NO_OUTPUT


@dataclass
class _TeamContext:
    """What every member of one team needs while it works on a task."""

    task: str
    genome: Genome
    gid: str
    inputs: Dict[str, Dict[str, Any]]
    img: Optional[str]
    episode: Optional[_Episode]


def _tokens(text: str) -> Set[str]:
    """Return the lowercase words in a text."""
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _jaccard(left: Set[str], right: Set[str]) -> float:
    """Return the word overlap of two word sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _survival(text: str, final: str, exclude: str = "") -> float:
    """Return the share of the final output's words a text introduced.

    Args:
        text: The text whose contribution is measured.
        final: The output the team ended with.
        exclude: Text whose words do not count as introduced.

    Returns:
        A number from 0 to 1.
    """
    final_words = _tokens(final)
    introduced = _tokens(text) - _tokens(exclude)
    return len(introduced & final_words) / max(1, len(final_words))


def _render(value: Any) -> str:
    """Render a structured message for an agent to read."""
    return json.dumps(
        value, indent=2, ensure_ascii=False, default=str
    )


def _first_tool_call(
    result: Any,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return the name and arguments of the first tool call in a run result.

    Args:
        result: What Agent.run returned: a list of tool calls, one tool
            call, or text when the model did not call a tool.

    Returns:
        The tool name and its arguments, or None and an empty dict.
    """
    calls = result if isinstance(result, list) else [result]

    for call in calls:
        if not isinstance(call, dict) and hasattr(call, "model_dump"):
            call = call.model_dump()

        function = (
            call.get("function")
            if isinstance(call, dict)
            else getattr(call, "function", None)
        )
        if function is None:
            continue

        if isinstance(function, dict):
            name, arguments = function.get("name"), function.get(
                "arguments"
            )
        else:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if name:
            return name, (
                arguments if isinstance(arguments, dict) else {}
            )

    return None, {}


def _to_float(value: Any, default: float) -> float:
    """Convert a value to a float, or return the default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(
    value: float, low: float = 0.0, high: float = 1.0
) -> float:
    """Clamp a number into a range."""
    return max(low, min(high, value))


def _normalise_kind(kind: Any) -> str:
    """Turn a free-text kind into a short, stable label."""
    label = re.sub(r"[^a-z0-9]+", "-", str(kind or "").lower())
    label = "-".join(label.strip("-").split("-")[:2]) or "general"
    return "synthesis" if label == INTEGRATE else label


def _normalise_shares(
    raw: Dict[str, float], members: List[str]
) -> Dict[str, float]:
    """Scale contribution scores so they sum to one over the members."""
    shares = {name: max(0.0, raw.get(name, 0.0)) for name in members}
    total = sum(shares.values())
    if total <= 0:
        return {name: 1.0 / len(members) for name in members}
    return {name: value / total for name, value in shares.items()}


def _sigmoid(value: float) -> float:
    """Return the logistic function of a number."""
    return 1.0 / (1.0 + math.exp(-value))


def _poisson(rng: random.Random, rate: float) -> int:
    """Draw a Poisson count with Knuth's method."""
    limit, count, product = math.exp(-rate), 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def _fiedler(
    names: List[str], weight: Callable[[str, str], float]
) -> Tuple[float, Dict[str, float]]:
    """Return a team's normalised connectivity and its Fiedler vector.

    Args:
        names: The team's members.
        weight: Edge weight from 0 to 1 between two members.

    Returns:
        The second-smallest Laplacian eigenvalue divided by team size,
        and each member's entry in its eigenvector.
    """
    size = len(names)
    edges = [
        [0.0 if i == j else weight(a, b) for j, b in enumerate(names)]
        for i, a in enumerate(names)
    ]
    degree = [sum(row) for row in edges]
    laplacian = [
        [
            (degree[i] if i == j else 0.0) - edges[i][j]
            for j in range(size)
        ]
        for i in range(size)
    ]
    shift = 2.0 * max(degree) + 1.0
    vector = [i - (size - 1) / 2.0 for i in range(size)]

    for _ in range(200):
        mean = sum(vector) / size
        vector = [value - mean for value in vector]
        vector = [
            shift * vector[i]
            - sum(laplacian[i][j] * vector[j] for j in range(size))
            for i in range(size)
        ]
        norm = (
            math.sqrt(sum(value * value for value in vector)) or 1.0
        )
        vector = [value / norm for value in vector]

    value = sum(
        vector[i] * laplacian[i][j] * vector[j]
        for i in range(size)
        for j in range(size)
    )
    return value / size, dict(zip(names, vector))


class SelfAssemblingSwarm:
    """A swarm whose agents form their own teams and search for the best architecture.

    Each epoch the agents assemble or repair a structure of teams,
    run it on the task, are scored by an independent verifier, receive
    credit, and reorganise: they join, leave and swap teams, and teams
    merge, split, change protocol or dissolve. The best structure seen
    is returned as the answer and kept as a reusable harness.

    Agents talk only through structured tool calls. At every step an
    agent gets a set of message tools through its LLM's
    tools_list_dictionary, must call one, and chooses which. While the
    swarm is calling an agent, the message tools stand in for the
    agent's own tools, MCP servers and streaming, and all of them are
    put back afterwards.

    The algorithm is described in SELF_ASSEMBLING_SWARM.md at the repo
    root.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "SelfAssemblingSwarm",
        description: str = "Agents form their own teams and search for the best architecture for a task.",
        agents: Optional[List[Agent]] = None,
        verifier: Optional[Callable[[str, str], float]] = None,
        judge_model_name: str = "gpt-5.4",
        max_loops: int = 6,
        max_subgoals: int = 5,
        team_size_cap: int = 4,
        teams_per_agent: int = 2,
        cost_weight: float = 0.2,
        latency_weight: float = 0.05,
        verifier_repeats: int = 1,
        budget_tokens: Optional[int] = None,
        target_score: Optional[float] = None,
        probe_size: int = 3,
        challenges: bool = True,
        peer_review: bool = True,
        max_workers: int = 8,
        config: Optional[CASConfig] = None,
        seed: Optional[int] = None,
        output_type: OutputType = "final",
        autosave: bool = False,
        verbose: bool = False,
    ):
        """Set up the swarm.

        Args:
            id: Unique id; generated when not given.
            name: Name used in logs and the conversation.
            description: What the swarm does.
            agents: The agents that organise themselves. Names must be unique.
            verifier: Scores an answer from 0 to 1 given the task and the
                answer. When not given, an LLM judge writes a rubric for
                the task, scores answers against it, and compares answers
                head to head to give each team credit.
            judge_model_name: Model of the default LLM judge.
            max_loops: Most epochs of search.
            max_subgoals: Most subgoals kept from the decomposition.
            team_size_cap: Most members on one team.
            teams_per_agent: Most teams one agent may be on at once.
            cost_weight: Weight of token cost against quality (lambda).
            latency_weight: Weight of time against quality (mu).
            verifier_repeats: Times each answer is scored; the mean is used.
            budget_tokens: Stop once this many tokens are spent in total.
            target_score: Stop once an epoch's quality reaches this.
            probe_size: Tasks sampled per epoch by run_family.
            challenges: Whether teams may challenge each other's outputs.
            peer_review: Whether team members rate each other for credit.
            max_workers: Most concurrent calls in one step.
            config: Tuning knobs; defaults when not given.
            seed: Seed for the swarm's random choices.
            output_type: How run formats the conversation.
            autosave: Save the conversation to the workspace after a run.
            verbose: Log each epoch and every structured message.

        Raises:
            ValueError: When the agents or settings are invalid.
        """
        self.id = id or generate_id("self-assembling-swarm")
        self.name = name
        self.description = description
        self.agents = agents
        self.verifier = verifier
        self.judge_model_name = judge_model_name
        self.max_loops = max_loops
        self.max_subgoals = max_subgoals
        self.team_size_cap = team_size_cap
        self.teams_per_agent = teams_per_agent
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.verifier_repeats = verifier_repeats
        self.budget_tokens = budget_tokens
        self.target_score = target_score
        self.probe_size = probe_size
        self.challenges = challenges
        self.peer_review = peer_review
        self.max_workers = max_workers
        self.config = config or CASConfig()
        self.seed = seed
        self.output_type = output_type
        self.autosave = autosave
        self.verbose = verbose

        self.reliability_check()

        self.agent_map: Dict[str, Agent] = {
            agent.agent_name: agent for agent in self.agents
        }
        self.names: List[str] = list(self.agent_map)
        self._locks: Dict[str, threading.Lock] = {
            agent_name: threading.Lock() for agent_name in self.names
        }

        self.verifier_agent: Optional[Agent] = None
        if verifier is None:
            self.verifier_agent = Agent(
                agent_name=f"{self.name}-Verifier",
                system_prompt=VERIFIER_PROMPT,
                model_name=judge_model_name,
                max_loops=1,
                print_on=False,
                persistent_memory=False,
                output_type="final",
            )
            self._locks[self.verifier_agent.agent_name] = (
                threading.Lock()
            )
        self._judge_lock = threading.Lock()

        self.workspace = WorkspaceManager(
            self,
            name=self.name or "self-assembling-swarm",
            verbose=verbose,
            enabled=autosave,
        )
        self.conversation = Conversation(time_enabled=False)
        self.harness: Optional[Dict[str, Any]] = None
        self.messages: List[Dict[str, Any]] = []
        self._reset_state()

        capture_init(self)

    def reliability_check(self) -> None:
        """Check the agents and settings before any call is made.

        Raises:
            ValueError: When the agents or settings are invalid.
        """
        if not self.agents:
            raise ValueError(
                "SelfAssemblingSwarm needs at least one agent."
            )

        names = [agent.agent_name for agent in self.agents]
        repeated = sorted(
            {
                agent_name
                for agent_name in names
                if names.count(agent_name) > 1
            }
        )
        if repeated:
            raise ValueError(
                f"Agent names must be unique; repeated: {repeated}. Teams, "
                "credit and trust are keyed by name."
            )

        for label, value in (
            ("max_loops", self.max_loops),
            ("max_subgoals", self.max_subgoals),
            ("team_size_cap", self.team_size_cap),
            ("teams_per_agent", self.teams_per_agent),
            ("verifier_repeats", self.verifier_repeats),
            ("probe_size", self.probe_size),
            ("max_workers", self.max_workers),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"{label} must be a positive integer."
                )

    def _reset_state(self) -> None:
        """Clear everything the swarm learns, so each search starts fresh."""
        self._rng = random.Random(self.seed)
        self.temperature = self.config.temperature
        self.skill: Dict[Tuple[str, str], float] = {}
        self.skill_count: Counter = Counter()
        self.trust: Dict[Tuple[str, str], float] = {}
        self.protocol_value: Dict[Tuple[str, str], float] = {}
        self.channel_value: Dict[Tuple[str, str], float] = {}
        self.baseline: Dict[str, float] = {}
        self.calibration: Dict[str, float] = {}
        self.ratings: Dict[Tuple[str, str], float] = {}
        self.familiar_kind: Counter = Counter()
        self.familiar_pair: Counter = Counter()
        self._fail: Counter = Counter()
        self._low: Counter = Counter()
        self._verify_cache: Dict[Tuple[str, str], float] = {}
        self._channel_credit: Dict[Tuple[str, str], float] = {}
        self._rubrics: Dict[str, List[str]] = {}
        self._compare_cache: Dict[Tuple[str, str, str], float] = {}
        self._next_id = 0
        self._epoch = 0
        self._token_unit: Optional[float] = None
        self._second_unit: Optional[float] = None
        self.archive: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.messages = []

    @trace_run(
        "SelfAssemblingSwarm.run",
        input_params=("task", "img"),
    )
    def run(self, task: str, img: Optional[str] = None) -> Any:
        """Search for the best architecture for one task and return its answer.

        Args:
            task: The task to complete.
            img: Image the working agents and the judge see with the task.

        Returns:
            The conversation formatted by output_type. With the default,
            "final", this is the best answer found.

        Raises:
            ValueError: When the task is empty.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string.")

        self.conversation = Conversation(time_enabled=False)
        self.conversation.add(role="user", content=task)

        best = self._search([task], img=img, family=False)

        self.conversation.add(
            role=self.name, content=best["answers"][0]
        )
        self.workspace.save_conversation(self.conversation)

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    @trace_run(
        "SelfAssemblingSwarm.run_family",
        input_params=("tasks", "img"),
    )
    def run_family(
        self, tasks: List[str], img: Optional[str] = None
    ) -> List[str]:
        """Search for one architecture that serves a family of similar tasks.

        Each epoch scores the structure on a fresh sample of probe_size
        tasks. The best harness then answers every task it was not
        already scored on.

        Args:
            tasks: Tasks of the same kind.
            img: Image every task is given with.

        Returns:
            One answer per task, in the order given.

        Raises:
            ValueError: When tasks is empty or holds an empty task.
        """
        if not tasks or not all(
            isinstance(task, str) and task.strip() for task in tasks
        ):
            raise ValueError(
                "tasks must be a list of non-empty strings."
            )

        self.conversation = Conversation(time_enabled=False)
        best = self._search(list(tasks), img=img, family=True)

        answers = dict(zip(best["tasks"], best["answers"]))
        unanswered = [
            task
            for task in dict.fromkeys(tasks)
            if task not in answers
        ]
        episodes = self._map(
            lambda task: self._episode(
                best["genome"], task, img, assess=False
            ),
            unanswered,
        )
        answers.update(
            {episode.task: episode.answer for episode in episodes}
        )

        for task in tasks:
            self.conversation.add(role="user", content=task)
            self.conversation.add(
                role=self.name, content=answers[task]
            )
        self.workspace.save_conversation(self.conversation)

        return [answers[task] for task in tasks]

    def run_harness(
        self,
        task: str,
        img: Optional[str] = None,
        harness: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run a frozen harness on a task, with no search and no scoring.

        Args:
            task: The task to complete.
            img: Image the working agents see with the task.
            harness: A harness from an earlier search; the latest one
                when not given.

        Returns:
            The conversation formatted by output_type.

        Raises:
            ValueError: When there is no harness, or it names an unknown agent.
        """
        harness = harness or self.harness
        if not harness:
            raise ValueError("Run a search first, or pass a harness.")

        genome = self._genome_from_harness(harness)
        episode = self._episode(genome, task, img, assess=False)

        self.conversation = Conversation(time_enabled=False)
        self.conversation.add(role="user", content=task)
        for gid in genome.order():
            self.conversation.add(
                role=self._team_label(genome, gid),
                content=episode.output(gid),
            )
        self.conversation.add(role=self.name, content=episode.answer)

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    @property
    def pareto(self) -> List[Dict[str, Any]]:
        """Archived structures that no other beats on both quality and tokens.

        Returns:
            Each entry's quality, tokens, fitness, epoch and harness.
        """
        entries = list(self.archive.values())
        front = [
            entry
            for entry in entries
            if not any(
                other["quality"] >= entry["quality"]
                and other["tokens"] <= entry["tokens"]
                and (
                    other["quality"] > entry["quality"]
                    or other["tokens"] < entry["tokens"]
                )
                for other in entries
            )
        ]
        return [
            {
                "quality": entry["quality"],
                "tokens": entry["tokens"],
                "fitness": entry["fitness"],
                "epoch": entry["epoch"],
                "harness": self._compile(entry),
            }
            for entry in sorted(
                front, key=lambda item: -item["quality"]
            )
        ]

    @property
    def report(self) -> Dict[str, Any]:
        """What the search did and learned.

        Returns:
            Per-epoch records, counts of each message type, and the
            learned skill, trust, protocol and channel estimates.
        """

        def keyed(
            table: Dict[Tuple[str, str], float],
        ) -> Dict[str, float]:
            return {
                " | ".join(key): round(value, 3)
                for key, value in sorted(
                    table.items(), key=lambda item: -item[1]
                )
            }

        return {
            "epochs": self.history,
            "messages": dict(
                Counter(message["type"] for message in self.messages)
            ),
            "temperature": self.temperature,
            "skill": keyed(self.skill),
            "trust": keyed(self.trust),
            "protocol_value": keyed(self.protocol_value),
            "channel_value": keyed(self.channel_value),
            "archive_cells": len(self.archive),
        }

    @property
    def usage(self) -> dict:
        """Provider token usage summed over every agent this swarm holds.

        Returns:
            A new dict with input_tokens, output_tokens, cached_tokens,
            reasoning_tokens and total_tokens.
        """
        return total_usage(self)

    def _search(
        self, tasks: List[str], img: Optional[str], family: bool
    ) -> Dict[str, Any]:
        """Run Coalition Architecture Search and return the best archive entry.

        Args:
            tasks: The task, or the family of tasks.
            img: Image every task is given with.
            family: Whether to sample probe tasks each epoch.

        Returns:
            The archive entry with the highest fitness.
        """
        self._reset_state()
        genome: Optional[Genome] = None
        previous: Optional[Set[Tuple[str, ...]]] = None
        best_fitness: List[float] = []
        unchanged = 0

        for epoch in range(1, self.max_loops + 1):
            self._epoch = epoch
            batch = (
                self._rng.sample(
                    tasks, min(self.probe_size, len(tasks))
                )
                if family
                else tasks
            )

            if genome is None:
                genome = self._assemble(tasks[0], img, generic=family)

            # No move follows the last epoch, so its credit would go unused.
            episodes = [
                self._episode(
                    genome, task, img, credit=epoch < self.max_loops
                )
                for task in batch
            ]
            entry = self._score(genome, episodes, epoch)
            self._learn(genome, episodes)
            self._record(genome, episodes, entry, family)

            best_fitness.append(
                max(item["fitness"] for item in self.archive.values())
            )
            signature = genome.signature()
            unchanged = unchanged + 1 if signature == previous else 0
            previous = signature

            if self._should_stop(
                epoch, entry, best_fitness, unchanged
            ):
                break

            genome, moves = self._adapt(genome, episodes, tasks[0])
            self.history[-1]["moves"] = moves

            others = [
                cell
                for cell in self.archive
                if cell != self._descriptor(genome, entry)
            ]
            if (
                others
                and self._rng.random() < self.config.restart_rate
            ):
                genome = deepcopy(
                    self.archive[self._rng.choice(others)]["genome"]
                )
                self.history[-1]["moves"].append(
                    "RESTART from archive"
                )

            self.temperature = max(
                self.config.min_temperature,
                self.temperature * self.config.temperature_decay,
            )

        best = max(
            self.archive.values(), key=lambda item: item["fitness"]
        )
        self.harness = self._compile(best)
        return best

    def _should_stop(
        self,
        epoch: int,
        entry: Dict[str, Any],
        best_fitness: List[float],
        unchanged: int,
    ) -> bool:
        """Return whether the search should end after this epoch."""
        patience = self.config.patience

        if (
            self.target_score is not None
            and entry["quality"] >= self.target_score
        ):
            return True
        if (
            self.budget_tokens is not None
            and self.usage["total_tokens"] >= self.budget_tokens
        ):
            return True
        if (
            epoch > patience
            and best_fitness[-1] - best_fitness[-1 - patience]
            < self.config.min_improvement
        ):
            return True
        return unchanged >= patience

    def _send(
        self,
        agent: Agent,
        prompt: str,
        palette: List[Dict[str, Any]],
        img: Optional[str] = None,
        team: Optional[str] = None,
        to: Optional[str] = None,
        episode: Optional[_Episode] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Have an agent answer by calling one tool from a palette.

        The palette is the agent's whole set of tools for this call and
        the model must call one of them. With one tool the choice names
        it; with several the model picks.

        Args:
            agent: The agent to call.
            prompt: What the agent is told.
            palette: The message tools it may use.
            img: Image sent with the prompt.
            team: The team the agent speaks for, for the message log.
            to: The team the message is for, for the message log.
            episode: The episode whose message log records it.

        Returns:
            The tool the agent chose and its arguments, or None and an
            empty dict when it called none.
        """
        names = [tool["function"]["name"] for tool in palette]

        with self._locks[agent.agent_name]:
            llm = getattr(agent, "llm", None)
            try:
                if isinstance(llm, LiteLLM):
                    with self._message_tools(agent, llm, palette):
                        result = agent.run(
                            task=prompt, img=img, messages=[]
                        )
                else:
                    result = agent.run(
                        task=prompt, img=img, messages=[]
                    )
                    result = agent_answer(agent, fallback=result)
            except Exception as error:
                logger.warning(
                    f"[{self.name}] {agent.agent_name} failed: "
                    f"{type(error).__name__}: {error}"
                )
                return None, {}

        name, arguments = _first_tool_call(result)
        if name not in names:
            name, arguments = None, {}
            # A model without forced tools may still answer in text.
            if "submit_output" in names and isinstance(result, str):
                name = "submit_output"
                arguments = {"output": result.strip(), "summary": ""}

        if name is None:
            logger.warning(
                f"[{self.name}] {agent.agent_name} called none of "
                f"{names}."
            )
            return None, {}

        if name == "request_input":
            to = str(arguments.get("team") or "") or to

        message = {
            "epoch": self._epoch,
            "type": name,
            "sender": agent.agent_name,
            "team": team,
            "to": to,
            "content": arguments,
        }
        self.messages.append(message)
        if episode is not None:
            episode.messages.append(message)
        if self.verbose:
            logger.info(
                f"[{self.name}] {agent.agent_name} ({team}) -> "
                f"{name}{' to ' + to if to else ''}: "
                f"{_render(arguments)[:300]}"
            )

        return name, arguments

    @contextmanager
    def _message_tools(
        self,
        agent: Agent,
        llm: LiteLLM,
        palette: List[Dict[str, Any]],
    ) -> Iterator[None]:
        """Swap an agent's tools for a message palette for one call."""
        agent_saved = {
            key: getattr(agent, key, None)
            for key in (
                "tools",
                "tool_loader",
                "max_loops",
                "output_type",
                "streaming_on",
            )
        }
        llm_saved = {
            key: getattr(llm, key, None)
            for key in (
                "tools_list_dictionary",
                "tool_choice",
                "stream",
            )
        }
        mcp_manager = agent.__dict__.get("mcp_manager", _MISSING)
        reasoning = llm.reasoning_effort or getattr(
            llm, "reasoning_enabled", None
        )

        agent.tools = None
        agent.tool_loader = None
        agent.max_loops = 1
        agent.output_type = "final"
        agent.streaming_on = False
        llm.tools_list_dictionary = palette
        llm.stream = False
        llm.tool_choice = (
            forced_tool_choice(
                palette[0],
                llm.reasoning_effort,
                getattr(llm, "reasoning_enabled", None),
            )
            if len(palette) == 1
            else "auto" if reasoning else "required"
        )
        # Agent runs MCP calls on any tool response, so the manager sits out this call.
        if mcp_manager is not _MISSING:
            agent.__dict__["mcp_manager"] = None

        try:
            yield
        finally:
            for key, value in agent_saved.items():
                setattr(agent, key, value)
            for key, value in llm_saved.items():
                setattr(llm, key, value)
            if mcp_manager is not _MISSING:
                agent.__dict__["mcp_manager"] = mcp_manager

    def _assemble(
        self, task: str, img: Optional[str], generic: bool
    ) -> Genome:
        """Build the first genome: decompose, self-assess, match and wire."""
        genome = Genome(subgoals=self._decompose(task, img, generic))
        genome.refresh()

        self._self_assess(task, genome)
        genome.teams = self._form_teams(genome)
        self._ensure_coverage(genome)
        self._wire(genome)

        return genome

    def _new_id(self) -> str:
        """Return a fresh subgoal id."""
        self._next_id += 1
        return f"s{self._next_id}"

    def _decompose(
        self, task: str, img: Optional[str], generic: bool
    ) -> Dict[str, Subgoal]:
        """Merge every agent's proposed decomposition into one subgoal graph."""
        prompt = DECOMPOSE_PROMPT.format(
            count=len(self.agents),
            task=task,
            max_subgoals=self._subgoal_limit(),
            generic=GENERIC_NOTE if generic else "",
        )
        proposals = self._map(
            lambda agent: self._send(
                agent,
                prompt,
                [PROPOSE_SUBGOALS, KEEP_WHOLE_TASK],
                img,
            ),
            self.agents,
        )

        whole = {
            "id": "whole",
            "description": "Complete the whole task.",
            "kind": "general",
            "depends_on": [],
            "output": "A complete answer to the task.",
            "support": 1,
        }
        drafts: List[Tuple[str, List[Dict[str, Any]]]] = []
        for agent, (tool, arguments) in zip(self.agents, proposals):
            if tool == "keep_whole_task":
                drafts.append((agent.agent_name, [whole]))
            elif tool == "propose_subgoals":
                items = [
                    item
                    for item in arguments.get("subgoals") or []
                    if isinstance(item, dict)
                    and str(item.get("description") or "").strip()
                ]
                drafts.append((agent.agent_name, items))

        items = self._consolidate(task, drafts, img) or self._cluster(
            drafts
        )
        return self._build_subgoals(items or [whole])

    def _consolidate(
        self,
        task: str,
        drafts: List[Tuple[str, List[Dict[str, Any]]]],
        img: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Have one randomly chosen agent merge every proposal into one list.

        Args:
            task: The task.
            drafts: Each agent's name and proposed subgoals.
            img: Image sent with the task.

        Returns:
            Canonical subgoals with support counts, or an empty list when
            the agent did not merge them.
        """
        if len(drafts) < 2:
            return []

        convener = self._rng.choice(self.agents)
        _, arguments = self._send(
            convener,
            CONSOLIDATE_PROMPT.format(
                task=task,
                proposals=_render(
                    [
                        {"from": name, "subgoals": items}
                        for name, items in drafts
                    ]
                ),
                limit=self._subgoal_limit(),
            ),
            [_consolidate_subgoals(self.names)],
            img,
        )

        merged = []
        for position, item in enumerate(
            arguments.get("subgoals") or []
        ):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description") or "").strip()
            if not description:
                continue
            supporters = {
                str(name)
                for name in item.get("proposed_by") or []
                if name in self.agent_map
            }
            merged.append(
                {
                    "id": str(item.get("id") or f"c{position}"),
                    "description": description,
                    "kind": _normalise_kind(item.get("kind")),
                    "output": str(item.get("output") or ""),
                    "depends_on": [
                        str(dep)
                        for dep in item.get("depends_on") or []
                    ],
                    "support": max(1, len(supporters)),
                }
            )
        return merged

    def _cluster(
        self, drafts: List[Tuple[str, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Merge proposals by word overlap, when no agent merged them.

        Args:
            drafts: Each agent's name and proposed subgoals.

        Returns:
            Canonical subgoals with support counts.
        """
        clusters: List[Dict[str, Any]] = []
        local_ids: Dict[Tuple[int, str], int] = {}
        pending: List[Tuple[int, int, List[str]]] = []

        for draft, (_, items) in enumerate(drafts):
            used: Set[int] = set()
            for position, item in enumerate(items):
                description = str(
                    item.get("description") or ""
                ).strip()
                words = _tokens(description)
                match, similarity = None, 0.0
                for index, cluster in enumerate(clusters):
                    overlap = _jaccard(words, cluster["words"])
                    if index not in used and overlap > similarity:
                        match, similarity = index, overlap

                if (
                    match is not None
                    and similarity >= self.config.duplicate_similarity
                ):
                    clusters[match]["support"] += 1
                else:
                    clusters.append(
                        {
                            "words": words,
                            "support": 1,
                            "description": description,
                            "kind": _normalise_kind(item.get("kind")),
                            "output": str(item.get("output") or ""),
                        }
                    )
                    match = len(clusters) - 1

                used.add(match)
                local_ids[
                    (draft, str(item.get("id") or position))
                ] = match
                pending.append(
                    (
                        match,
                        draft,
                        [
                            str(dep)
                            for dep in item.get("depends_on") or []
                        ],
                    )
                )

        depends: Dict[int, Set[str]] = {}
        for index, draft, deps in pending:
            for dep in deps:
                target = local_ids.get((draft, dep))
                if target is not None and target != index:
                    depends.setdefault(index, set()).add(f"c{target}")

        return [
            {
                "id": f"c{index}",
                "description": cluster["description"],
                "kind": cluster["kind"],
                "output": cluster["output"],
                "depends_on": sorted(depends.get(index, set())),
                "support": cluster["support"],
            }
            for index, cluster in enumerate(clusters)
        ]

    def _build_subgoals(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Subgoal]:
        """Keep the best-supported subgoals and wire their dependencies.

        Args:
            items: Canonical subgoals with ids, dependencies and support.

        Returns:
            Subgoals by id, with the integrate subgoal added.
        """
        min_support = min(self.config.min_support, len(self.agents))
        ranked = sorted(items, key=lambda item: -item["support"])
        kept = [
            item
            for item in ranked
            if item["support"] >= min_support
            or self._rng.random() < self.temperature / 2
        ][: self._subgoal_limit()]
        if not kept:
            kept = ranked[:1]

        subgoals: Dict[str, Subgoal] = {}
        ids: Dict[str, str] = {}
        for item in kept:
            gid = self._new_id()
            ids[item["id"]] = gid
            subgoals[gid] = Subgoal(
                id=gid,
                description=item["description"],
                kind=item["kind"],
                output=item["output"],
                support=item["support"],
            )

        subgoals[INTEGRATE] = Subgoal(
            id=INTEGRATE,
            description="Combine the other teams' outputs into the complete final answer to the overall task.",
            kind=INTEGRATE,
            output="The final answer to the overall task.",
        )
        trial = Genome(subgoals=subgoals)

        for item in kept:
            subgoal = subgoals[ids[item["id"]]]
            for dep in item["depends_on"]:
                target = ids.get(dep)
                if target is None or target == subgoal.id:
                    continue
                subgoal.depends_on.append(target)
                if trial.has_cycle():
                    subgoal.depends_on.pop()

        return subgoals

    def _self_assess(self, task: str, genome: Genome) -> None:
        """Have every agent rate its own fit for every subgoal."""
        ids = list(genome.subgoals)
        listing = _render(
            [
                {
                    "subgoal": subgoal.id,
                    "kind": subgoal.kind,
                    "description": subgoal.description,
                }
                for subgoal in genome.subgoals.values()
            ]
        )
        prompt = ASSESS_PROMPT.format(task=task, subgoals=listing)
        replies = self._map(
            lambda agent: self._send(
                agent, prompt, [_rate_subgoals(ids)]
            ),
            self.agents,
        )

        for agent_name, (_, arguments) in zip(self.names, replies):
            scores = {
                str(item.get("subgoal")): item.get("score")
                for item in arguments.get("ratings") or []
                if isinstance(item, dict)
            }
            for gid in ids:
                self.ratings[(agent_name, gid)] = _clamp(
                    _to_float(scores.get(gid), 0.5)
                )

    def _fit(
        self, agent_name: str, gid: str, genome: Genome
    ) -> float:
        """Return an agent's fit for a subgoal: self-rating blended with record."""
        kind = genome.subgoals[gid].kind
        evidence = self.skill_count[(agent_name, kind)]
        alpha = 1.0 / (1.0 + evidence / self.config.self_report_decay)

        return alpha * self.ratings.get((agent_name, gid), 0.5) + (
            1.0 - alpha
        ) * self.skill.get((agent_name, kind), 0.5)

    def _trust(self, left: str, right: str) -> float:
        """Return how well two agents have done together, from -1 to 1."""
        return self.trust.get(tuple(sorted((left, right))), 0.0)

    def _utility(
        self,
        agent_name: str,
        gid: str,
        genome: Genome,
        members: List[str],
        load: int,
    ) -> float:
        """Return what an agent gains from being on a subgoal's team."""
        others = [name for name in members if name != agent_name]
        kind = genome.subgoals[gid].kind
        trust = (
            sum(self._trust(agent_name, name) for name in others)
            / len(others)
            if others
            else 0.0
        )
        familiarity = self.familiar_kind[(agent_name, kind)] + (
            sum(
                self.familiar_pair[tuple(sorted((agent_name, name)))]
                for name in others
            )
            / len(others)
            if others
            else 0.0
        )

        return (
            self._fit(agent_name, gid, genome)
            + self.config.trust_weight * trust
            + self.config.novelty_weight
            * self.temperature
            / math.sqrt(1.0 + familiarity)
            - self.config.load_weight * load
        )

    def _choose(self, options: List[Any], values: List[float]) -> Any:
        """Pick an option with probability softmax(value / temperature)."""
        scale = max(self.temperature, 1e-6)
        top = max(values)
        weights = [
            math.exp((value - top) / scale) for value in values
        ]
        return self._rng.choices(options, weights=weights, k=1)[0]

    def _sample_protocol(
        self, kind: str, size: int, solo: bool = True
    ) -> str:
        """Sample a protocol a team of this size can run, from what has worked."""
        feasible = [
            protocol
            for protocol in PROTOCOLS
            if PROTOCOL_MIN_SIZE[protocol] <= max(1, size)
            and (solo or protocol != "SOLO")
        ]
        return self._choose(
            feasible,
            [
                self.protocol_value.get((protocol, kind), 0.0)
                for protocol in feasible
            ],
        )

    def _form_teams(self, genome: Genome) -> Dict[str, Team]:
        """Turn charters into teams through deferred acceptance."""
        charters: List[Dict[str, Any]] = []

        for gid, subgoal in genome.subgoals.items():
            ranked = sorted(
                self.names,
                key=lambda name: -self._fit(name, gid, genome),
            )
            for founder in ranked[: self.config.founders_per_subgoal]:
                charters.append(
                    {
                        "gid": gid,
                        "founder": founder,
                        "fixed": [],
                        "protocol": self._sample_protocol(
                            subgoal.kind, self.team_size_cap
                        ),
                    }
                )

        capacity = {name: self.teams_per_agent for name in self.names}
        held = self._deferred_acceptance(genome, charters, capacity)

        teams: Dict[str, Team] = {}
        for gid in genome.subgoals:
            options = [
                index
                for index, charter in enumerate(charters)
                if charter["gid"] == gid and held[index]
            ]
            if not options:
                teams[gid] = Team(subgoal=gid)
                continue
            winner = max(
                options,
                key=lambda index: sum(
                    self._utility(name, gid, genome, held[index], 0)
                    for name in held[index]
                ),
            )
            teams[gid] = Team(
                subgoal=gid,
                members=list(held[winner]),
                protocol=charters[winner]["protocol"],
            )

        load = Counter(
            name for team in teams.values() for name in team.members
        )
        refill = [
            {
                "gid": gid,
                "founder": None,
                "fixed": list(team.members),
                "protocol": team.protocol,
            }
            for gid, team in teams.items()
        ]
        spare = {
            name: self.teams_per_agent - load[name]
            for name in self.names
        }
        for charter, members in zip(
            refill, self._deferred_acceptance(genome, refill, spare)
        ):
            teams[charter["gid"]].members = members

        return teams

    def _charter_cap(self, charter: Dict[str, Any]) -> int:
        """Return how many members a charter may hold."""
        if charter["protocol"] == "SOLO":
            return max(1, len(charter["fixed"]))
        return self.team_size_cap

    def _deferred_acceptance(
        self,
        genome: Genome,
        charters: List[Dict[str, Any]],
        capacity: Dict[str, int],
    ) -> List[List[str]]:
        """Match agents to charters so no agent and charter prefer each other.

        Args:
            genome: The genome whose subgoals the charters are for.
            charters: Each with gid, founder, fixed members and protocol.
            capacity: How many more teams each agent may join.

        Returns:
            The members each charter ends with, fixed members included.
        """
        held = [list(charter["fixed"]) for charter in charters]

        def anchor(index: int) -> List[str]:
            charter = charters[index]
            return charter["fixed"] or (
                [charter["founder"]] if charter["founder"] else []
            )

        def charter_rank(index: int, agent_name: str) -> float:
            gid = charters[index]["gid"]
            others = [
                name for name in anchor(index) if name != agent_name
            ]
            trust = (
                sum(self._trust(agent_name, name) for name in others)
                / len(others)
                if others
                else 0.0
            )
            founder = (
                1.0
                if agent_name == charters[index]["founder"]
                else 0.0
            )
            return (
                founder
                + self._fit(agent_name, gid, genome)
                + self.config.trust_weight * trust
            )

        preferences = {
            name: sorted(
                [
                    index
                    for index, charter in enumerate(charters)
                    if name not in charter["fixed"]
                ],
                key=lambda index: -self._utility(
                    name,
                    charters[index]["gid"],
                    genome,
                    anchor(index),
                    0,
                ),
            )
            for name in self.names
            if capacity.get(name, 0) > 0
        }
        cursor: Counter = Counter()
        holding: Counter = Counter()
        free = list(preferences)

        while free:
            name = free.pop(0)
            while holding[name] < capacity[name] and cursor[
                name
            ] < len(preferences[name]):
                index = preferences[name][cursor[name]]
                cursor[name] += 1
                gid = charters[index]["gid"]

                same_subgoal = any(
                    name in held[other]
                    for other, charter in enumerate(charters)
                    if charter["gid"] == gid
                )
                utility = self._utility(
                    name, gid, genome, anchor(index), holding[name]
                )
                if same_subgoal or (
                    holding[name] > 0
                    and utility < self.config.join_threshold
                ):
                    continue

                held[index].append(name)
                holding[name] += 1

                if len(held[index]) > self._charter_cap(
                    charters[index]
                ):
                    removable = [
                        member
                        for member in held[index]
                        if member not in charters[index]["fixed"]
                    ]
                    worst = min(
                        removable,
                        key=lambda member: charter_rank(
                            index, member
                        ),
                    )
                    held[index].remove(worst)
                    holding[worst] -= 1
                    if worst != name and worst not in free:
                        free.append(worst)

        return held

    def _subgoal_limit(self) -> int:
        """Return the most work subgoals the agents have seats for."""
        seats = len(self.agents) * self.teams_per_agent
        return max(1, min(self.max_subgoals, seats - 1))

    def _free_seat(
        self, genome: Genome, gid: str, load: Counter
    ) -> str:
        """Pick an agent for an empty team without breaking the seat cap.

        Takes an agent with a free seat when there is one. Otherwise moves
        the best fit off a team that has more than one member.

        Args:
            genome: The genome being repaired.
            gid: The subgoal whose team is empty.
            load: How many teams each agent is on, updated in place.

        Returns:
            The agent's name.
        """
        open_agents = [
            name
            for name in self.names
            if load[name] < self.teams_per_agent
        ]
        if open_agents:
            pick = max(
                open_agents,
                key=lambda name: (
                    -load[name],
                    self._fit(name, gid, genome),
                ),
            )
            load[pick] += 1
            return pick

        donors = [
            (name, other)
            for other, team in genome.teams.items()
            if len(team.members) > 1
            for name in team.members
        ]
        if donors:
            name, other = max(
                donors,
                key=lambda pair: self._fit(pair[0], gid, genome)
                - self._fit(pair[0], pair[1], genome),
            )
            genome.teams[other].members.remove(name)
            return name

        pick = min(self.names, key=lambda name: load[name])
        load[pick] += 1
        return pick

    def _ensure_coverage(self, genome: Genome) -> None:
        """Give every subgoal a team, order members by fit and fix protocols."""
        for gid in genome.subgoals:
            genome.teams.setdefault(gid, Team(subgoal=gid))

        for team in genome.teams.values():
            team.members = [
                name
                for name in dict.fromkeys(team.members)
                if name in self.agent_map
            ]

        load = Counter(
            name
            for team in genome.teams.values()
            for name in team.members
        )
        for gid, team in genome.teams.items():
            if not team.members:
                team.members = [self._free_seat(genome, gid, load)]

        for gid, team in genome.teams.items():
            team.members.sort(
                key=lambda name: -self._fit(name, gid, genome)
            )
            size = len(team.members)
            if PROTOCOL_MIN_SIZE.get(team.protocol, 99) > size or (
                team.protocol == "SOLO" and size > 1
            ):
                team.protocol = self._sample_protocol(
                    genome.subgoals[gid].kind, size, solo=size == 1
                )

    def _wire(self, genome: Genome) -> None:
        """Add learned and random channels between teams."""
        gids = list(genome.subgoals)
        linked = {
            tuple(sorted((dep, gid)))
            for gid in gids
            for dep in genome.subgoals[gid].depends_on
        }
        for pair in linked:
            genome.channels.pop(pair, None)

        pairs = [
            tuple(sorted((left, right)))
            for position, left in enumerate(gids)
            for right in gids[position + 1 :]
        ]
        pairs = [
            pair
            for pair in pairs
            if pair not in linked and pair not in genome.channels
        ]

        for pair in pairs:
            kinds = tuple(
                sorted(genome.subgoals[gid].kind for gid in pair)
            )
            chance = _sigmoid(
                self.channel_value.get(kinds, 0.0)
                + self.config.channel_bias
            )
            if self._rng.random() < chance:
                genome.channels[pair] = "learned"

        open_pairs = [
            pair for pair in pairs if pair not in genome.channels
        ]
        count = _poisson(
            self._rng,
            self.config.random_channel_rate * self.temperature,
        )
        for pair in self._rng.sample(
            open_pairs, min(count, len(open_pairs))
        ):
            genome.channels[pair] = "random"

    def _map(
        self, function: Callable[[Any], Any], items: List[Any]
    ) -> List[Any]:
        """Apply a function to items concurrently, keeping their order."""
        items = list(items)
        if len(items) <= 1:
            return [function(item) for item in items]

        with ContextThreadPoolExecutor(
            max_workers=min(self.max_workers, len(items))
        ) as pool:
            return list(pool.map(function, items))

    def _artifact_message(
        self, genome: Genome, gid: str, artifact: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return a team's submitted output as the message other teams read."""
        return {
            "from_team": gid,
            "subgoal": genome.subgoals[gid].description,
            "output": artifact.get("output") or NO_OUTPUT,
            "summary": artifact.get("summary", ""),
            "confidence": artifact.get("confidence"),
            "assumptions": artifact.get("assumptions", []),
        }

    def _opening(self, context: _TeamContext) -> str:
        """Build the shared opening of every prompt a team's members get."""
        genome, gid = context.genome, context.gid
        subgoal = genome.subgoals[gid]
        roster = [
            {
                "team": other,
                "subgoal": genome.subgoals[other].description,
                "members": genome.teams[other].members,
            }
            for other in genome.teams
            if other != gid
        ]
        sections = [
            f"Overall task:\n{context.task}",
            (
                f"You are on team {gid}, which owns this subgoal: "
                f"{subgoal.description}\nIt must produce: "
                f"{subgoal.output or 'what the subgoal describes'}"
            ),
            f"The other teams in the swarm:\n{_render(roster)}",
        ]
        if context.inputs:
            sections.append(
                "Messages from the teams whose work you build on:\n"
                + _render(
                    [
                        self._artifact_message(genome, dep, artifact)
                        for dep, artifact in context.inputs.items()
                    ]
                )
            )
        return "\n\n".join(sections)

    def _work(
        self, context: _TeamContext, agent: Agent, instruction: str
    ) -> Dict[str, Any]:
        """Run one working step: the agent submits output or asks a team first.

        Args:
            context: The team and task the agent is working on.
            agent: The member doing the step.
            instruction: What this step asks of it.

        Returns:
            The submitted output, summary, confidence and assumptions.
        """
        prompt = f"{self._opening(context)}\n\n{instruction}"
        episode = context.episode
        # Only a team that has submitted output can answer a question.
        others = [
            gid
            for gid in (episode.artifacts if episode else {})
            if gid != context.gid
        ]

        for asked in range(self.config.requests_per_step + 1):
            palette = [SUBMIT_OUTPUT]
            if (
                others
                and episode is not None
                and asked < self.config.requests_per_step
            ):
                palette.append(_request_input(others))

            tool, arguments = self._send(
                agent,
                prompt,
                palette,
                context.img,
                team=context.gid,
                episode=episode,
            )
            if tool == "request_input":
                reply = self._answer_request(context, arguments)
                prompt += "\n\n" + REQUEST_REPLY.format(
                    reply=_render(reply)
                )
                continue
            if tool == "submit_output":
                return {
                    "output": str(
                        arguments.get("output") or ""
                    ).strip(),
                    "summary": str(arguments.get("summary") or ""),
                    "confidence": _clamp(
                        _to_float(arguments.get("confidence"), 0.5)
                    ),
                    "assumptions": [
                        str(item)
                        for item in arguments.get("assumptions") or []
                    ],
                }
            break

        return {
            "output": "",
            "summary": "",
            "confidence": 0.0,
            "assumptions": [],
        }

    def _answer_request(
        self, context: _TeamContext, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route a question to another team's lead and return its reply."""
        genome, asker = context.genome, context.gid
        target = str(arguments.get("team") or "")
        question = str(arguments.get("question") or "").strip()
        if (
            target not in genome.teams
            or target == asker
            or not question
        ):
            return {
                "from_team": target,
                "answered": False,
                "reply": "No such team.",
            }

        episode = context.episode
        current = ""
        if episode is not None and target in episode.artifacts:
            current = (
                "Your team's current output:\n"
                + _render(
                    self._artifact_message(
                        genome, target, episode.artifacts[target]
                    )
                )
                + "\n"
            )

        lead = self.agent_map[genome.teams[target].members[0]]
        tool, reply = self._send(
            lead,
            REQUEST_PROMPT.format(
                task=context.task,
                own_id=target,
                own=genome.subgoals[target].description,
                current=current,
                asker_id=asker,
                asker=genome.subgoals[asker].description,
                question=question,
            ),
            [ANSWER_REQUEST, DECLINE_REQUEST],
            team=target,
            to=asker,
            episode=episode,
        )
        if episode is not None:
            episode.traffic[(asker, target)] += 1
            episode.traffic[(target, asker)] += 1
            episode.requests[tuple(sorted((asker, target)))] += 1

        return {
            "from_team": target,
            "answered": tool == "answer_request",
            "reply": str(
                reply.get("answer") or reply.get("reason") or ""
            ),
        }

    def _run_team(
        self,
        task: str,
        genome: Genome,
        gid: str,
        inputs: Dict[str, Dict[str, Any]],
        img: Optional[str],
        episode: Optional[_Episode],
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """Run one team's protocol on its subgoal.

        Returns:
            The submitted artifact, each member's share of it, the trace
            of who submitted what, and the number of internal turns.
        """
        team = genome.teams[gid]
        members = [self.agent_map[name] for name in team.members]
        context = _TeamContext(
            task, genome, gid, inputs, img, episode
        )

        runner = getattr(self, f"_protocol_{team.protocol.lower()}")
        artifact, raw, trace, turns = runner(context, members)
        artifact = dict(artifact)
        artifact["output"] = artifact.get("output") or NO_OUTPUT

        return (
            artifact,
            _normalise_shares(raw, team.members),
            trace,
            turns,
        )

    def _protocol_solo(
        self, context: _TeamContext, members: List[Agent]
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """One agent does the subgoal."""
        lead = members[0]
        artifact = self._work(context, lead, DRAFT)
        trace = [(lead.agent_name, artifact["output"])]
        return artifact, {lead.agent_name: 1.0}, trace, 1

    def _protocol_chain(
        self, context: _TeamContext, members: List[Agent]
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """Each member refines the last one's output; the best fit goes last."""
        steps: List[Tuple[str, Dict[str, Any]]] = []

        for agent in reversed(members):
            if steps:
                instruction = IMPROVE.format(
                    draft=_render(
                        {"from": steps[-1][0], **steps[-1][1]}
                    )
                )
            else:
                instruction = DRAFT
            artifact = self._work(context, agent, instruction)
            if not artifact["output"] and steps:
                artifact = steps[-1][1]
            steps.append((agent.agent_name, artifact))

        final = steps[-1][1]["output"]
        own: Dict[str, float] = {}
        previous = ""
        for agent_name, artifact in steps:
            own[agent_name] = _survival(
                artifact["output"], final, exclude=previous
            )
            previous = artifact["output"]

        trace = [
            (name, artifact["output"]) for name, artifact in steps
        ]
        return steps[-1][1], own, trace, len(steps)

    def _protocol_fanout(
        self, context: _TeamContext, members: List[Agent]
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """Members work independently and the lead merges their drafts."""
        drafts = self._map(
            lambda agent: self._work(context, agent, DRAFT), members
        )
        names = [agent.agent_name for agent in members]
        merger = members[0]

        merged = self._work(
            context,
            merger,
            MERGE.format(
                drafts=_render(
                    [
                        {"from": name, **draft}
                        for name, draft in zip(names, drafts)
                    ]
                )
            ),
        )
        if not merged["output"]:
            merged = max(
                drafts, key=lambda draft: len(draft["output"])
            )
        output = merged["output"]

        own = {
            name: _survival(draft["output"], output)
            for name, draft in zip(names, drafts)
        }
        own[merger.agent_name] += _survival(
            output,
            output,
            exclude=" ".join(draft["output"] for draft in drafts),
        )
        trace = [
            (name, draft["output"])
            for name, draft in zip(names, drafts)
        ] + [(merger.agent_name, output)]

        return merged, own, trace, len(members) + 1

    def _protocol_critic_loop(
        self, context: _TeamContext, members: List[Agent]
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """The lead drafts, the others review, and the lead revises."""
        maker, critics = members[0], members[1:]
        draft = self._work(context, maker, DRAFT)
        first = draft["output"]
        trace = [(maker.agent_name, first)]
        turns = 1
        revisions: List[Tuple[List[str], str, str]] = []
        opening = self._opening(context)

        for _ in range(self.config.critic_rounds):
            review = f"{opening}\n\n" + CRITIQUE.format(
                draft=_render({"from": maker.agent_name, **draft})
            )
            verdicts = self._map(
                lambda critic, review=review: self._send(
                    critic,
                    review,
                    [APPROVE_DRAFT, REQUEST_CHANGES],
                    context.img,
                    team=context.gid,
                    episode=context.episode,
                ),
                critics,
            )
            turns += len(critics)
            changes = [
                {
                    "from": critic.agent_name,
                    "issues": arguments.get("issues") or [],
                    "critique": arguments.get("critique", ""),
                }
                for critic, (tool, arguments) in zip(
                    critics, verdicts
                )
                if tool == "request_changes"
            ]
            if not changes:
                break

            trace.extend(
                (change["from"], _render(change["issues"]))
                for change in changes
            )
            revised = self._work(
                context,
                maker,
                REVISE.format(
                    draft=_render(draft), critiques=_render(changes)
                ),
            )
            if not revised["output"]:
                revised = draft
            turns += 1
            revisions.append(
                (
                    [change["from"] for change in changes],
                    draft["output"],
                    revised["output"],
                )
            )
            trace.append((maker.agent_name, revised["output"]))
            draft = revised

        own = {agent.agent_name: 0.0 for agent in members}
        own[maker.agent_name] = _survival(first, draft["output"])
        for names, before, after in revisions:
            gain = _survival(after, draft["output"], exclude=before)
            for agent_name in names:
                own[agent_name] += gain / len(names)

        return draft, own, trace, turns

    def _protocol_debate(
        self, context: _TeamContext, members: List[Agent]
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """Members argue positions over rounds and the lead judges."""
        judge, sides = members[0], members[1:]
        positions = self._map(
            lambda agent: self._work(context, agent, DRAFT), sides
        )
        turns = len(sides)
        names = [agent.agent_name for agent in sides]

        for _ in range(self.config.debate_rounds - 1):

            def rebut(
                position: int,
                positions: List[Dict[str, Any]] = positions,
            ) -> Dict[str, Any]:
                others = [
                    {"from": names[other], **positions[other]}
                    for other in range(len(sides))
                    if other != position
                ]
                revised = self._work(
                    context,
                    sides[position],
                    REBUT.format(
                        own=_render(positions[position]),
                        others=_render(others),
                    ),
                )
                return (
                    revised
                    if revised["output"]
                    else positions[position]
                )

            positions = self._map(rebut, list(range(len(sides))))
            turns += len(sides)

        verdict = self._work(
            context,
            judge,
            JUDGE.format(
                positions=_render(
                    [
                        {"from": name, **position}
                        for name, position in zip(names, positions)
                    ]
                )
            ),
        )
        if not verdict["output"]:
            verdict = max(
                positions, key=lambda item: len(item["output"])
            )
        output = verdict["output"]

        own = {
            name: _survival(position["output"], output)
            for name, position in zip(names, positions)
        }
        own[judge.agent_name] = _survival(
            output,
            output,
            exclude=" ".join(
                position["output"] for position in positions
            ),
        )
        trace = [
            (name, position["output"])
            for name, position in zip(names, positions)
        ] + [(judge.agent_name, output)]

        return verdict, own, trace, turns + 1

    def _protocol_vote(
        self, context: _TeamContext, members: List[Agent]
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """Members answer independently, then vote on the anonymous answers."""
        answers = self._map(
            lambda agent: self._work(context, agent, DRAFT), members
        )
        labels = [
            chr(ord("A") + index) for index in range(len(members))
        ]
        ballot = f"{self._opening(context)}\n\n" + BALLOT.format(
            candidates=_render(
                [
                    {"candidate": label, **answer}
                    for label, answer in zip(labels, answers)
                ]
            )
        )

        ballots = self._map(
            lambda agent: self._send(
                agent,
                ballot,
                [_cast_vote(labels)],
                context.img,
                team=context.gid,
                episode=context.episode,
            ),
            members,
        )
        votes = Counter(
            str(arguments.get("candidate") or "")
            for _, arguments in ballots
        )
        winner = max(
            range(len(members)),
            key=lambda index: (votes[labels[index]], -index),
        )
        output = answers[winner]["output"]

        names = [agent.agent_name for agent in members]
        own = {
            name: _survival(answer["output"], output)
            for name, answer in zip(names, answers)
        }
        trace = [
            (name, answer["output"])
            for name, answer in zip(names, answers)
        ]
        return answers[winner], own, trace, 2 * len(members)

    def _protocol_lead(
        self, context: _TeamContext, members: List[Agent]
    ) -> Tuple[
        Dict[str, Any], Dict[str, float], List[Tuple[str, str]], int
    ]:
        """The lead hands out parts and assembles them, or does it alone."""
        lead, workers = members[0], members[1:]
        names = [agent.agent_name for agent in workers]

        tool, arguments = self._send(
            lead,
            f"{self._opening(context)}\n\n"
            + PLAN.format(names=", ".join(names)),
            [_assign_parts(names), SUBMIT_OUTPUT],
            context.img,
            team=context.gid,
            episode=context.episode,
        )
        if tool == "submit_output" and arguments.get("output"):
            artifact = {
                "output": str(arguments.get("output")).strip(),
                "summary": str(arguments.get("summary") or ""),
                "confidence": _clamp(
                    _to_float(arguments.get("confidence"), 0.5)
                ),
                "assumptions": list(
                    arguments.get("assumptions") or []
                ),
            }
            trace = [(lead.agent_name, artifact["output"])]
            return artifact, {lead.agent_name: 1.0}, trace, 1

        assignments = {
            str(item.get("agent")): str(item.get("part") or "")
            for item in arguments.get("assignments") or []
            if isinstance(item, dict)
        }
        parts = self._map(
            lambda agent: self._work(
                context,
                agent,
                PART.format(
                    part=assignments.get(agent.agent_name)
                    or "The whole subgoal."
                ),
            ),
            workers,
        )
        assembled = self._work(
            context,
            lead,
            ASSEMBLE.format(
                parts=_render(
                    [
                        {"from": name, **part}
                        for name, part in zip(names, parts)
                    ]
                )
            ),
        )
        if not assembled["output"]:
            assembled = {
                **parts[0],
                "output": "\n\n".join(
                    part["output"] for part in parts
                ),
            }
        output = assembled["output"]

        own = {
            name: _survival(part["output"], output)
            for name, part in zip(names, parts)
        }
        own[lead.agent_name] = _survival(
            output,
            output,
            exclude=" ".join(part["output"] for part in parts),
        )
        trace = [
            (name, part["output"]) for name, part in zip(names, parts)
        ] + [(lead.agent_name, output)]

        return assembled, own, trace, len(workers) + 2

    def _episode(
        self,
        genome: Genome,
        task: str,
        img: Optional[str],
        assess: bool = True,
        credit: bool = True,
    ) -> _Episode:
        """Run a genome on one task, then score it and assign credit.

        Args:
            genome: The structure to run.
            task: The task.
            img: Image the working agents see with the task.
            assess: Whether to score the answer.
            credit: Whether to measure each team's credit after scoring.

        Returns:
            The episode's outputs, messages, score and credit.
        """
        episode = _Episode(task=task, epoch=self._epoch)
        tokens_before = self._agent_tokens()
        started = time.monotonic()

        for wave in genome.waves():
            results = self._map(
                lambda gid: self._run_team(
                    task,
                    genome,
                    gid,
                    {
                        dep: episode.artifacts[dep]
                        for dep in genome.subgoals[gid].depends_on
                    },
                    img,
                    episode,
                ),
                wave,
            )

            for gid, (artifact, shares, trace, turns) in zip(
                wave, results
            ):
                episode.artifacts[gid] = artifact
                episode.shares[gid] = shares
                episode.traces[gid] = trace
                episode.traffic[(gid, gid)] += turns
                for dep in genome.subgoals[gid].depends_on:
                    episode.traffic[(dep, gid)] += 1

            self._exchange(genome, task, wave, episode, img)

        episode.answer = episode.output(INTEGRATE)
        episode.tokens = self._agent_tokens() - tokens_before
        episode.seconds = time.monotonic() - started

        if assess:
            episode.quality = self._verify(task, episode.answer, img)
        if assess and credit:
            self._assign_credit(genome, task, img, episode)

        return episode

    def _exchange(
        self,
        genome: Genome,
        task: str,
        wave: List[str],
        episode: _Episode,
        img: Optional[str],
    ) -> None:
        """Let linked teams challenge or endorse this wave's outputs."""
        if not self.challenges:
            return

        pairs: List[Tuple[str, str]] = []
        for target in wave:
            for sender in genome.challengers(target):
                if (
                    episode.sent[sender]
                    < self.config.challenge_budget
                ):
                    pairs.append((sender, target))
                    episode.sent[sender] += 1
        if not pairs:
            return

        def review(
            pair: Tuple[str, str],
        ) -> Tuple[Optional[str], Dict[str, Any]]:
            sender, target = pair
            lead = self.agent_map[genome.teams[sender].members[0]]
            return self._send(
                lead,
                CHALLENGE.format(
                    task=task,
                    own_id=sender,
                    own=genome.subgoals[sender].description,
                    other_id=target,
                    other=genome.subgoals[target].description,
                    artifact=_render(
                        self._artifact_message(
                            genome, target, episode.artifacts[target]
                        )
                    ),
                ),
                [RAISE_CHALLENGE, ENDORSE_OUTPUT],
                team=sender,
                to=target,
                episode=episode,
            )

        raised: Dict[str, List[Dict[str, Any]]] = {}
        for (sender, target), (tool, arguments) in zip(
            pairs, self._map(review, pairs)
        ):
            if tool == "raise_challenge":
                episode.traffic[(sender, target)] += 1
                raised.setdefault(target, []).append(
                    {"from_team": sender, **arguments}
                )

        def answer(
            target: str,
        ) -> Tuple[Optional[str], Dict[str, Any]]:
            lead = self.agent_map[genome.teams[target].members[0]]
            inputs = {
                dep: episode.artifacts[dep]
                for dep in genome.subgoals[target].depends_on
            }
            context = _TeamContext(
                task, genome, target, inputs, img, episode
            )
            prompt = (
                f"{self._opening(context)}\n\n"
                + ANSWER_CHALLENGES.format(
                    artifact=_render(episode.artifacts[target]),
                    challenges=_render(raised[target]),
                )
            )
            return self._send(
                lead,
                prompt,
                [REVISE_OUTPUT, STAND_BY_OUTPUT],
                img,
                team=target,
                episode=episode,
            )

        targets = list(raised)
        for target, (tool, arguments) in zip(
            targets, self._map(answer, targets)
        ):
            senders = [item["from_team"] for item in raised[target]]
            episode.challenged[target] = senders
            episode.traffic[(target, target)] += 1
            for sender in senders:
                episode.traffic[(target, sender)] += 1

            revised = str(arguments.get("output") or "").strip()
            if tool == "revise_output" and revised:
                episode.pre_revision[target] = episode.artifacts[
                    target
                ]
                episode.artifacts[target] = {
                    **episode.artifacts[target],
                    "output": revised,
                }

    def _lead_integrate(
        self,
        genome: Genome,
        task: str,
        artifacts: Dict[str, Dict[str, Any]],
        img: Optional[str],
    ) -> str:
        """Have the integrate lead combine a set of leaf outputs on its own."""
        lead = self.agent_map[genome.teams[INTEGRATE].members[0]]
        inputs = {
            dep: artifacts[dep]
            for dep in genome.subgoals[INTEGRATE].depends_on
        }
        context = _TeamContext(
            task, genome, INTEGRATE, inputs, img, None
        )
        tool, arguments = self._send(
            lead,
            f"{self._opening(context)}\n\n{DRAFT}",
            [SUBMIT_OUTPUT],
            img,
            team=INTEGRATE,
        )
        return str(arguments.get("output") or "") if tool else ""

    def _assign_credit(
        self,
        genome: Genome,
        task: str,
        img: Optional[str],
        episode: _Episode,
    ) -> None:
        """Measure team credit by ablation, then split it among members."""
        leaves = genome.subgoals[INTEGRATE].depends_on
        missing = {
            "output": NO_OUTPUT,
            "summary": "",
            "assumptions": [],
        }
        variants: List[Dict[str, Dict[str, Any]]] = [
            dict(episode.artifacts)
        ]
        for gid in leaves:
            variants.append({**episode.artifacts, gid: missing})

        sampled = [
            target
            for target in episode.pre_revision
            if target in leaves
        ][: self.config.sampled_channels]
        for target in sampled:
            variants.append(
                {
                    **episode.artifacts,
                    target: episode.pre_revision[target],
                }
            )

        answers = self._map(
            lambda artifacts: self._lead_integrate(
                genome, task, artifacts, img
            ),
            variants,
        )
        reference = answers[0]
        concatenated = "\n\n".join(
            episode.output(gid) for gid in leaves
        )

        # Head to head: does the answer get worse without this team?
        pairs = [(reference, answer) for answer in answers[1:]]
        pairs.append((episode.answer, concatenated))
        if INTEGRATE in episode.pre_revision:
            pairs.append(
                (
                    episode.answer,
                    episode.pre_revision[INTEGRATE]["output"],
                )
            )
        margins = self._map(
            lambda pair: self._compare(task, pair[0], pair[1], img),
            pairs,
        )

        count = len(leaves)
        credit = dict(zip(leaves, margins[:count]))
        credit[INTEGRATE] = margins[count + len(sampled)]

        for target, margin in zip(
            sampled, margins[count : count + len(sampled)]
        ):
            senders = episode.challenged[target]
            for sender in senders:
                episode.channel_credit[(sender, target)] = (
                    margin / len(senders)
                )
        if INTEGRATE in episode.pre_revision:
            senders = episode.challenged[INTEGRATE]
            for sender in senders:
                episode.channel_credit[(sender, INTEGRATE)] = margins[
                    -1
                ] / len(senders)

        for gid in reversed(genome.order()):
            if gid in credit:
                continue
            credit[gid] = self.config.upstream_share * sum(
                credit[dep] / len(genome.subgoals[dep].depends_on)
                for dep in genome.dependents(gid)
                if dep != INTEGRATE
            )
        episode.team_credit = credit

        if self.peer_review:
            self._collect_reviews(genome, task, episode)

        for gid, team in genome.teams.items():
            weights = self._member_weights(team, episode)
            for agent_name, weight in weights.items():
                episode.agent_credit[(agent_name, gid)] = (
                    credit[gid] * weight
                )

    def _collect_reviews(
        self, genome: Genome, task: str, episode: _Episode
    ) -> None:
        """Have each member of a multi-member team rate its teammates."""
        jobs = [
            (gid, agent_name)
            for gid, team in genome.teams.items()
            if len(team.members) > 1
            for agent_name in team.members
        ]

        def review(job: Tuple[str, str]) -> Dict[str, float]:
            gid, rater = job
            teammates = [
                name
                for name in genome.teams[gid].members
                if name != rater
            ]
            trace = _render(
                [
                    {"from": name, "submitted": text[:600]}
                    for name, text in episode.traces[gid]
                ]
            )
            _, arguments = self._send(
                self.agent_map[rater],
                PEER_REVIEW.format(
                    task=task,
                    subgoal=genome.subgoals[gid].description,
                    trace=trace,
                ),
                [_rate_teammates(teammates)],
                team=gid,
                episode=episode,
            )
            return {
                str(item.get("agent")): _clamp(
                    _to_float(item.get("score"), 0.5)
                )
                for item in arguments.get("ratings") or []
                if isinstance(item, dict)
                and item.get("agent") in teammates
            }

        for (gid, rater), ratings in zip(
            jobs, self._map(review, jobs)
        ):
            if ratings:
                episode.ratings.setdefault(gid, {})[rater] = ratings

    def _member_weights(
        self, team: Team, episode: _Episode
    ) -> Dict[str, float]:
        """Blend measured contribution with calibrated peer ratings."""
        own = episode.shares.get(team.subgoal) or _normalise_shares(
            {}, team.members
        )
        ratings = episode.ratings.get(team.subgoal, {})
        if not ratings:
            return own

        peer: Dict[str, float] = {}
        for agent_name in team.members:
            weighted = [
                (self.calibration.get(rater, 0.5), given[agent_name])
                for rater, given in ratings.items()
                if agent_name in given
            ]
            total = sum(weight for weight, _ in weighted)
            peer[agent_name] = (
                sum(weight * score for weight, score in weighted)
                / total
                if total
                else own[agent_name]
            )

        peer = _normalise_shares(peer, team.members)
        theta = self.config.own_weight
        return _normalise_shares(
            {
                agent_name: theta * own[agent_name]
                + (1 - theta) * peer[agent_name]
                for agent_name in team.members
            },
            team.members,
        )

    def _verify(
        self, task: str, answer: str, img: Optional[str]
    ) -> float:
        """Score an answer from 0 to 1 with the verifier, cached per answer."""
        if not str(answer or "").strip() or answer == NO_OUTPUT:
            return 0.0

        key = (task, answer)
        if key in self._verify_cache:
            return self._verify_cache[key]

        scores = []
        for _ in range(self.verifier_repeats):
            if self.verifier is not None:
                scores.append(
                    _clamp(float(self.verifier(task, answer)))
                )
                continue
            rubric = self._rubric(task, img)
            if not rubric:
                _, arguments = self._send(
                    self.verifier_agent,
                    VERIFY_TASK.format(task=task, answer=answer),
                    [SCORE_ANSWER],
                    img,
                )
                scores.append(
                    _clamp(
                        _to_float(arguments.get("score"), 0.0) / 10.0
                    )
                )
                continue

            _, arguments = self._send(
                self.verifier_agent,
                RUBRIC_SCORE_PROMPT.format(
                    task=task,
                    criteria=_render(
                        [
                            {"criterion": number, "text": text}
                            for number, text in enumerate(rubric, 1)
                        ]
                    ),
                    answer=answer,
                ),
                [_score_rubric(len(rubric))],
                img,
            )
            given = {
                int(_to_float(item.get("criterion"), 0)): _clamp(
                    _to_float(item.get("score"), 0.0), 0.0, 3.0
                )
                for item in arguments.get("scores") or []
                if isinstance(item, dict)
            }
            scores.append(
                sum(
                    given.get(number, 0.0)
                    for number in range(1, len(rubric) + 1)
                )
                / (3 * len(rubric))
            )

        score = sum(scores) / len(scores)
        self._verify_cache[key] = score
        return score

    def _rubric(self, task: str, img: Optional[str]) -> List[str]:
        """Return the judge's rubric for a task, writing it on first use."""
        with self._judge_lock:
            if task not in self._rubrics:
                _, arguments = self._send(
                    self.verifier_agent,
                    RUBRIC_PROMPT.format(task=task),
                    [WRITE_RUBRIC],
                    img,
                )
                self._rubrics[task] = [
                    str(criterion).strip()
                    for criterion in arguments.get("criteria") or []
                    if str(criterion).strip()
                ][:10]
            return self._rubrics[task]

    def _compare(
        self, task: str, base: str, variant: str, img: Optional[str]
    ) -> float:
        """Return how much better one answer is than another, from -1 to 1.

        A callable verifier gives the difference of its two scores. The
        LLM judge sees both answers in random order and says which is
        better and by how much.

        Args:
            task: The task both answers are for.
            base: The answer credit is measured from.
            variant: The answer with one team's work removed or changed.
            img: Image sent with the task.

        Returns:
            Positive when base is better, negative when variant is.
        """
        if base == variant:
            return 0.0
        if self.verifier is not None:
            return self._verify(task, base, img) - self._verify(
                task, variant, img
            )
        if not str(variant).strip() or variant == NO_OUTPUT:
            return 1.0
        if not str(base).strip() or base == NO_OUTPUT:
            return -1.0

        key = (task, base, variant)
        if key in self._compare_cache:
            return self._compare_cache[key]

        margins = []
        for _ in range(self.verifier_repeats):
            flipped = self._rng.random() < 0.5
            first, second = (
                (variant, base) if flipped else (base, variant)
            )
            _, arguments = self._send(
                self.verifier_agent,
                COMPARE_PROMPT.format(
                    task=task, first=first, second=second
                ),
                [COMPARE_ANSWERS],
                img,
            )
            better = arguments.get("better")
            margin = _clamp(_to_float(arguments.get("margin"), 0.0))
            if better not in ("first", "second"):
                margins.append(0.0)
                continue
            base_won = (better == "first") != flipped
            margins.append(margin if base_won else -margin)

        value = sum(margins) / len(margins)
        self._compare_cache[key] = value
        return value

    def _agent_tokens(self) -> int:
        """Return the tokens the working agents have spent so far."""
        return total_usage(*self.agents)["total_tokens"]

    def _score(
        self, genome: Genome, episodes: List[_Episode], epoch: int
    ) -> Dict[str, Any]:
        """Compute fitness for an epoch and keep the genome in the archive."""
        count = len(episodes)
        quality = sum(episode.quality for episode in episodes) / count
        tokens = sum(episode.tokens for episode in episodes) / count
        seconds = sum(episode.seconds for episode in episodes) / count

        self._token_unit = self._token_unit or max(tokens, 1.0)
        self._second_unit = self._second_unit or max(seconds, 1e-6)
        cost = tokens / self._token_unit
        fitness = (
            quality
            - self.cost_weight * cost
            - self.latency_weight * seconds / self._second_unit
        )

        entry = {
            "fitness": fitness,
            "quality": quality,
            "tokens": tokens,
            "seconds": seconds,
            "cost": cost,
            "epoch": epoch,
            "genome": deepcopy(genome),
            "tasks": [episode.task for episode in episodes],
            "answers": [episode.answer for episode in episodes],
            "traffic": sum(
                total
                for episode in episodes
                for (left, right), total in episode.traffic.items()
                if left != right
            )
            / count,
        }

        cell = self._descriptor(genome, entry)
        if (
            cell not in self.archive
            or fitness > self.archive[cell]["fitness"]
        ):
            self.archive[cell] = entry
        return entry

    def _descriptor(
        self, genome: Genome, entry: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Return the archive cell of a structure: size, depth, chatter and cost."""
        teams = len(genome.teams)
        depth = len(genome.waves())
        traffic = entry["traffic"] / max(1, teams)
        cost = entry["cost"]

        return (
            (
                0
                if teams <= 1
                else 1 if teams <= 3 else 2 if teams <= 6 else 3
            ),
            min(depth, 4),
            0 if traffic < 1 else 1 if traffic < 3 else 2,
            (
                0
                if cost < 0.75
                else 1 if cost < 1.25 else 2 if cost < 2 else 3
            ),
        )

    def _learn(
        self, genome: Genome, episodes: List[_Episode]
    ) -> None:
        """Update skill, trust, protocol, channel and calibration estimates."""
        rate = self.config.learning_rate
        gain = self.config.credit_gain

        def mean(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def blend(
            table: Dict[Any, float],
            key: Any,
            target: float,
            start: float,
        ) -> None:
            table[key] = (1 - rate) * table.get(
                key, start
            ) + rate * target

        for gid, team in genome.teams.items():
            kind = genome.subgoals[gid].kind
            delta = mean(
                [
                    episode.team_credit.get(gid, 0.0)
                    for episode in episodes
                ]
            )
            advantage = delta - self.baseline.get(kind, 0.0)
            blend(self.baseline, kind, delta, 0.0)

            for agent_name in team.members:
                credit = mean(
                    [
                        episode.agent_credit.get(
                            (agent_name, gid), 0.0
                        )
                        for episode in episodes
                    ]
                )
                performance = 0.5 + 0.5 * math.tanh(
                    gain * credit * len(team.members)
                )
                blend(
                    self.skill, (agent_name, kind), performance, 0.5
                )
                self.skill_count[(agent_name, kind)] += 1
                self.familiar_kind[(agent_name, kind)] += 1

            for position, left in enumerate(team.members):
                for right in team.members[position + 1 :]:
                    pair = tuple(sorted((left, right)))
                    blend(
                        self.trust,
                        pair,
                        math.tanh(gain * advantage),
                        0.0,
                    )
                    self.familiar_pair[pair] += 1

            blend(
                self.protocol_value,
                (team.protocol, kind),
                math.tanh(gain * advantage),
                0.0,
            )

            team_quality = 0.5 + 0.5 * math.tanh(gain * delta)
            for episode in episodes:
                for rater, given in episode.ratings.get(
                    gid, {}
                ).items():
                    blend(
                        self.calibration,
                        rater,
                        1.0
                        - abs(
                            mean(list(given.values())) - team_quality
                        ),
                        0.5,
                    )

            self._fail[gid] = self._fail[gid] + 1 if delta <= 0 else 0
            self._low[gid] = (
                self._low[gid] + 1
                if delta < self.config.low_share
                else 0
            )

        channel_credit: Dict[Tuple[str, str], List[float]] = {}
        for episode in episodes:
            for pair, value in episode.channel_credit.items():
                channel_credit.setdefault(pair, []).append(value)
        self._channel_credit = {
            tuple(sorted(pair)): mean(values)
            for pair, values in channel_credit.items()
        }
        for (sender, target), values in channel_credit.items():
            kinds = tuple(
                sorted(
                    (
                        genome.subgoals[sender].kind,
                        genome.subgoals[target].kind,
                    )
                )
            )
            blend(
                self.channel_value,
                kinds,
                math.tanh(gain * mean(values)),
                0.0,
            )

    def _adapt(
        self, genome: Genome, episodes: List[_Episode], task: str
    ) -> Tuple[Genome, List[str]]:
        """Apply team moves and agent moves to a copy of the genome."""
        genome = deepcopy(genome)
        traffic: Counter = Counter()
        for episode in episodes:
            traffic.update(episode.traffic)

        moves = self._team_moves(genome, traffic, task)
        moves += self._agent_moves(genome)

        used = {
            tuple(sorted((sender, target)))
            for episode in episodes
            for target, senders in episode.challenged.items()
            for sender in senders
        }
        asked = {
            pair for episode in episodes for pair in episode.requests
        }
        dependencies = {
            tuple(sorted((dep, gid)))
            for gid, subgoal in genome.subgoals.items()
            for dep in subgoal.depends_on
        }

        for pair in list(genome.channels):
            kept = pair in used | asked
            if not kept or self._channel_credit.get(pair, 0.0) < 0:
                del genome.channels[pair]
                moves.append(f"UNLINK {pair[0]}-{pair[1]}")

        # Teams that chose to ask each other questions get a channel.
        for pair in sorted(
            asked - dependencies - set(genome.channels)
        ):
            if (
                pair[0] in genome.subgoals
                and pair[1] in genome.subgoals
            ):
                genome.channels[pair] = "learned"
                moves.append(f"LINK {pair[0]}-{pair[1]}")

        genome.refresh()
        self._ensure_coverage(genome)
        self._wire(genome)

        return genome, moves

    def _team_moves(
        self, genome: Genome, traffic: Counter, task: str
    ) -> List[str]:
        """Dissolve, merge, split and mutate teams from measured signals."""
        moves: List[str] = []

        for gid in list(genome.work_ids()):
            if self._fail[gid] < self.config.fail_epochs:
                continue
            self._fail[gid] = 0
            consumers = [
                dep
                for dep in genome.dependents(gid)
                if dep != INTEGRATE
            ]
            if not consumers and len(genome.work_ids()) > 1:
                del genome.subgoals[gid]
                genome.refresh()
                moves.append(f"DISSOLVE {gid} (dropped)")
            else:
                genome.teams[gid].members = []
                moves.append(f"DISSOLVE {gid} (re-staffed)")

        candidates = []
        work = genome.work_ids()
        for position, left in enumerate(work):
            for right in work[position + 1 :]:
                cross = (
                    traffic[(left, right)] + traffic[(right, left)]
                )
                inside = (
                    traffic[(left, left)] + traffic[(right, right)]
                )
                members = set(genome.teams[left].members) | set(
                    genome.teams[right].members
                )
                ratio = cross / max(1, inside)
                if (
                    ratio > self.config.merge_threshold
                    and len(members) <= self.team_size_cap
                ):
                    candidates.append((ratio, left, right))

        merged: Set[str] = set()
        for _, left, right in sorted(candidates, reverse=True):
            if left in merged or right in merged:
                continue
            new_id = self._merge(genome, left, right)
            if new_id:
                merged.update((left, right))
                moves.append(f"MERGE {left}+{right} -> {new_id}")

        for gid in genome.work_ids():
            team = genome.teams[gid]
            if not team.members:
                continue
            weak = False
            if len(team.members) > 2:
                connectivity, _ = _fiedler(
                    team.members,
                    lambda left, right: (self._trust(left, right) + 1)
                    / 2,
                )
                weak = connectivity < self.config.split_threshold
            failing = self._low[gid] >= self.config.fail_epochs
            seats = len(self.agents) * self.teams_per_agent
            if (weak or failing) and len(genome.teams) < seats:
                new_ids = self._split(
                    genome, gid, task, by_trust=weak
                )
                if new_ids:
                    moves.append(
                        f"SPLIT {gid} -> {'+'.join(new_ids)}"
                    )
                    break

        for gid, team in genome.teams.items():
            if (
                self._rng.random()
                >= self.config.mutation_rate * self.temperature
            ):
                continue
            protocol = self._sample_protocol(
                genome.subgoals[gid].kind, len(team.members)
            )
            if protocol != team.protocol:
                moves.append(
                    f"MUTATE {gid} {team.protocol} -> {protocol}"
                )
                team.protocol = protocol

        return moves

    def _merge(
        self, genome: Genome, left: str, right: str
    ) -> Optional[str]:
        """Merge two subgoals and their teams; undo it if it makes a cycle."""
        trial = deepcopy(genome)
        first, second = trial.subgoals[left], trial.subgoals[right]
        new_id = self._new_id()
        major = first if first.support >= second.support else second

        trial.subgoals[new_id] = Subgoal(
            id=new_id,
            description=f"{first.description} Also: {second.description}",
            kind=major.kind,
            output=f"{first.output}; {second.output}".strip("; "),
            depends_on=[
                dep
                for dep in first.depends_on + second.depends_on
                if dep not in (left, right)
            ],
            support=first.support + second.support,
        )
        for subgoal in trial.subgoals.values():
            if (
                left in subgoal.depends_on
                or right in subgoal.depends_on
            ):
                subgoal.depends_on = [
                    new_id if dep in (left, right) else dep
                    for dep in subgoal.depends_on
                ]

        teams = (trial.teams[left], trial.teams[right])
        larger = max(teams, key=lambda team: len(team.members))
        trial.teams[new_id] = Team(
            subgoal=new_id,
            members=list(
                dict.fromkeys(teams[0].members + teams[1].members)
            ),
            protocol=larger.protocol,
        )
        for name in self.names:
            self.ratings[(name, new_id)] = max(
                self.ratings.get((name, left), 0.5),
                self.ratings.get((name, right), 0.5),
            )

        del trial.subgoals[left], trial.subgoals[right]
        trial.refresh()
        if trial.has_cycle():
            return None

        genome.subgoals = trial.subgoals
        genome.teams = trial.teams
        genome.channels = trial.channels
        return new_id

    def _split(
        self, genome: Genome, gid: str, task: str, by_trust: bool
    ) -> Optional[List[str]]:
        """Ask the team lead to split its subgoal in two, and split the team."""
        subgoal, team = genome.subgoals[gid], genome.teams[gid]
        lead = self.agent_map[team.members[0]]
        tool, arguments = self._send(
            lead,
            SPLIT_PROMPT.format(
                task=task,
                subgoal=subgoal.description,
                output=subgoal.output,
            ),
            [SPLIT_SUBGOAL, KEEP_SUBGOAL],
            team=gid,
        )
        parts = [
            part
            for part in arguments.get("parts") or []
            if isinstance(part, dict)
            and str(part.get("description") or "").strip()
        ]
        if tool != "split_subgoal" or len(parts) < 2:
            return None

        if by_trust:
            _, vector = _fiedler(
                team.members,
                lambda left, right: (self._trust(left, right) + 1)
                / 2,
            )
            groups = [
                [name for name in team.members if vector[name] >= 0],
                [name for name in team.members if vector[name] < 0],
            ]
        else:
            groups = [team.members[0::2], team.members[1::2]]

        new_ids = [self._new_id(), self._new_id()]
        for new_id, part, members in zip(new_ids, parts, groups):
            genome.subgoals[new_id] = Subgoal(
                id=new_id,
                description=str(part["description"]).strip(),
                kind=subgoal.kind,
                output=str(part.get("output") or ""),
                depends_on=list(subgoal.depends_on),
                support=subgoal.support,
            )
            genome.teams[new_id] = Team(
                subgoal=new_id,
                members=list(members),
                protocol=team.protocol,
            )
            for name in self.names:
                self.ratings[(name, new_id)] = self.ratings.get(
                    (name, gid), 0.5
                )

        for other in genome.subgoals.values():
            if gid in other.depends_on:
                other.depends_on = [
                    dep for dep in other.depends_on if dep != gid
                ] + new_ids

        del genome.subgoals[gid]
        genome.refresh()
        return new_ids

    def _agent_moves(self, genome: Genome) -> List[str]:
        """Let up to a capped share of agents stay, leave, join, move or swap."""
        cap, limit = self.team_size_cap, self.teams_per_agent
        load = Counter(
            name
            for team in genome.teams.values()
            for name in team.members
        )
        proposals = []

        for agent_name in self.names:
            current = [
                gid
                for gid, team in genome.teams.items()
                if agent_name in team.members
            ]
            options: List[Tuple[Any, ...]] = [("STAY",)]
            gains: List[float] = [0.0]

            for here in current:
                team = genome.teams[here]
                stay = self._utility(
                    agent_name,
                    here,
                    genome,
                    team.members,
                    load[agent_name] - 1,
                )
                if len(team.members) > 1:
                    options.append(("LEAVE", here))
                    gains.append(-stay)

                for there, other in genome.teams.items():
                    if agent_name in other.members:
                        continue
                    if (
                        len(other.members) < cap
                        and len(team.members) > 1
                    ):
                        options.append(("MOVE", here, there))
                        gains.append(
                            self._utility(
                                agent_name,
                                there,
                                genome,
                                other.members + [agent_name],
                                load[agent_name] - 1,
                            )
                            - stay
                        )
                    for partner in other.members:
                        if partner in team.members:
                            continue
                        partner_gain = self._utility(
                            partner,
                            here,
                            genome,
                            [
                                m
                                for m in team.members
                                if m != agent_name
                            ]
                            + [partner],
                            load[partner] - 1,
                        ) - self._utility(
                            partner,
                            there,
                            genome,
                            other.members,
                            load[partner] - 1,
                        )
                        if partner_gain < 0:
                            continue
                        options.append(("SWAP", here, there, partner))
                        gains.append(
                            self._utility(
                                agent_name,
                                there,
                                genome,
                                [
                                    m
                                    for m in other.members
                                    if m != partner
                                ]
                                + [agent_name],
                                load[agent_name] - 1,
                            )
                            - stay
                            + partner_gain
                        )

            if load[agent_name] < limit:
                for there, other in genome.teams.items():
                    if (
                        agent_name not in other.members
                        and len(other.members) < cap
                    ):
                        options.append(("JOIN", there))
                        gains.append(
                            self._utility(
                                agent_name,
                                there,
                                genome,
                                other.members + [agent_name],
                                load[agent_name],
                            )
                        )

            choice = self._choose(options, gains)
            if choice[0] != "STAY":
                proposals.append(
                    (
                        max(abs(gain) for gain in gains),
                        agent_name,
                        choice,
                    )
                )

        movers = math.ceil(
            self.config.move_fraction * len(self.names)
        )
        moves = []
        for _, agent_name, choice in sorted(proposals, reverse=True)[
            :movers
        ]:
            if self._apply_move(genome, agent_name, choice):
                moves.append(
                    f"{choice[0]} {agent_name} {' '.join(choice[1:])}"
                )
        return moves

    def _apply_move(
        self, genome: Genome, agent_name: str, choice: Tuple[Any, ...]
    ) -> bool:
        """Apply one agent move if it is still valid; return whether it was."""
        teams, cap = genome.teams, self.team_size_cap
        kind = choice[0]

        if kind == "LEAVE":
            team = teams.get(choice[1])
            if (
                not team
                or agent_name not in team.members
                or len(team.members) < 2
            ):
                return False
            team.members.remove(agent_name)
            return True

        if kind == "JOIN":
            team = teams.get(choice[1])
            if (
                not team
                or agent_name in team.members
                or len(team.members) >= cap
            ):
                return False
            team.members.append(agent_name)
            return True

        if kind == "MOVE":
            here, there = teams.get(choice[1]), teams.get(choice[2])
            if (
                not here
                or not there
                or agent_name not in here.members
                or agent_name in there.members
                or len(here.members) < 2
                or len(there.members) >= cap
            ):
                return False
            here.members.remove(agent_name)
            there.members.append(agent_name)
            return True

        here, there = teams.get(choice[1]), teams.get(choice[2])
        partner = choice[3]
        if (
            not here
            or not there
            or agent_name not in here.members
            or partner not in there.members
            or partner in here.members
            or agent_name in there.members
        ):
            return False
        here.members[here.members.index(agent_name)] = partner
        there.members[there.members.index(partner)] = agent_name
        return True

    def _team_label(self, genome: Genome, gid: str) -> str:
        """Return a readable name for a team."""
        team = genome.teams[gid]
        return (
            f"{gid} [{genome.subgoals[gid].kind}, {team.protocol}: "
            f"{', '.join(team.members)}]"
        )

    def _record(
        self,
        genome: Genome,
        episodes: List[_Episode],
        entry: Dict[str, Any],
        family: bool,
    ) -> None:
        """Log an epoch in the history and the conversation."""
        epoch = entry["epoch"]
        structure = [
            self._team_label(genome, gid) for gid in genome.teams
        ]
        credit = {
            gid: round(
                sum(
                    episode.team_credit.get(gid, 0.0)
                    for episode in episodes
                )
                / len(episodes),
                3,
            )
            for gid in genome.teams
        }
        messages = Counter(
            message["type"]
            for episode in episodes
            for message in episode.messages
        )
        self.history.append(
            {
                "epoch": epoch,
                "quality": round(entry["quality"], 3),
                "fitness": round(entry["fitness"], 3),
                "tokens": entry["tokens"],
                "seconds": round(entry["seconds"], 2),
                "temperature": round(self.temperature, 3),
                "structure": structure,
                "team_credit": credit,
                "messages": dict(messages),
                "moves": [],
            }
        )

        summary = (
            f"Epoch {epoch}: quality {entry['quality']:.2f}, fitness "
            f"{entry['fitness']:.2f}, {len(genome.teams)} teams\n"
            + "\n".join(f"- {line}" for line in structure)
        )
        if not family:
            summary += f"\n\n{episodes[0].answer}"
        self.conversation.add(
            role=f"{self.name} epoch {epoch}", content=summary
        )

        if self.verbose:
            logger.info(f"[{self.name}] {summary.splitlines()[0]}")

    def _compile(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Freeze an archived genome into a serialisable harness spec."""
        genome: Genome = entry["genome"]
        roles = {
            "SOLO": ("worker", "worker"),
            "CHAIN": ("final step", "step"),
            "FANOUT": ("merger", "drafter"),
            "CRITIC_LOOP": ("maker", "critic"),
            "DEBATE": ("judge", "side"),
            "VOTE": ("voter", "voter"),
            "LEAD": ("lead", "worker"),
        }
        teams = []
        for gid in genome.order() + [INTEGRATE]:
            team = genome.teams[gid]
            first, rest = roles[team.protocol]
            teams.append(
                {
                    "subgoal": gid,
                    "protocol": team.protocol,
                    "members": [
                        {
                            "name": name,
                            "role": first if index == 0 else rest,
                        }
                        for index, name in enumerate(team.members)
                    ],
                }
            )

        load = Counter(
            name
            for team in genome.teams.values()
            for name in team.members
        )
        return {
            "name": self.name,
            "algorithm": "Coalition Architecture Search",
            "fitness": entry["fitness"],
            "quality": entry["quality"],
            "tokens": entry["tokens"],
            "epoch": entry["epoch"],
            "subgoals": [
                asdict(subgoal)
                for subgoal in genome.subgoals.values()
            ],
            "teams": teams,
            "channels": [
                {"from": dep, "to": gid, "kind": "dependency"}
                for gid, subgoal in genome.subgoals.items()
                for dep in subgoal.depends_on
            ]
            + [
                {"between": list(pair), "kind": kind}
                for pair, kind in genome.channels.items()
            ],
            "liaisons": [
                {
                    "agent": name,
                    "teams": [
                        gid
                        for gid, team in genome.teams.items()
                        if name in team.members
                    ],
                }
                for name, count in load.items()
                if count > 1
            ],
        }

    def _genome_from_harness(self, harness: Dict[str, Any]) -> Genome:
        """Rebuild a genome from a harness spec."""
        subgoals = {
            item["id"]: Subgoal(**item)
            for item in harness["subgoals"]
        }
        teams = {}
        for item in harness["teams"]:
            names = [member["name"] for member in item["members"]]
            unknown = [
                name for name in names if name not in self.agent_map
            ]
            if unknown:
                raise ValueError(
                    f"The harness names unknown agents: {unknown}"
                )
            teams[item["subgoal"]] = Team(
                subgoal=item["subgoal"],
                members=names,
                protocol=item["protocol"],
            )
        channels = {
            tuple(sorted(item["between"])): item["kind"]
            for item in harness["channels"]
            if "between" in item
        }

        genome = Genome(
            subgoals=subgoals, teams=teams, channels=channels
        )
        genome.refresh()
        return genome
