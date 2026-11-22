"""TeamFormationSwarm: agents bid on subtasks, form teams around them, and share results across teams."""

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from swarms.structs.agent import Agent, reset_agents
from swarms.structs.context_utils import agent_answer
from swarms.structs.conversation import Conversation
from swarms.structs.usage_utils import total_usage
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)
from swarms.tools.tool_choice import forced_tool_choice
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM
from swarms.utils.output_types import OutputType

PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "post_subtasks",
        "description": "Post the subtasks that teams will work on.",
        "parameters": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One short instruction per subtask.",
                },
            },
            "required": ["subtasks"],
        },
    },
}

BID_TOOL = {
    "type": "function",
    "function": {
        "name": "bid",
        "description": "Bid for the subtask you want to work on.",
        "parameters": {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "integer",
                    "description": "The number of the subtask you choose.",
                },
                "score": {
                    "type": "number",
                    "description": "How strongly you want it, 0 to 1.",
                    "minimum": 0,
                    "maximum": 1,
                },
                "reason": {
                    "type": "string",
                    "description": "One line on why you fit it.",
                },
            },
            "required": ["subtask", "score", "reason"],
        },
    },
}

COORDINATOR_PROMPT = (
    "You coordinate a swarm of agents. You split tasks into subtasks "
    "that teams can work on at the same time, and you combine the "
    "teams' results into one answer."
)

PLAN_PROMPT = """Split this task into at most {max_subtasks} subtasks that separate teams can work on at the same time. Call post_subtasks with one short instruction per subtask.

Task: {task}"""

BID_PROMPT = """The task below has been split into subtasks. Choose the one subtask where your skills help most and say how strongly you want it, from 0 to 1. Call bid with its number, your score and a one-line reason.

Task: {task}

Subtasks:
{subtasks}"""

TEAM_TURN_PROMPT = """Overall task: {task}

You are on {team} with {teammates}. Your team's subtask: {subtask}

Results the teams have shared so far:
{board}

Your team's discussion this round:
{discussion}

{instruction}"""

MEMBER_INSTRUCTION = (
    "Add your contribution to your team's subtask. Build on your "
    "teammates and use other teams' results where they help."
)

LEADER_INSTRUCTION = (
    "You lead this team. Write the team's result for its subtask, "
    "combining your teammates' contributions. Every other team will "
    "read it."
)

SYNTHESIS_PROMPT = """Combine the teams' results into one complete answer to the task.

Task: {task}

Team results:
{results}"""


@dataclass
class Team:
    """A subtask and the agents that chose it, highest bidder first.

    Args:
        name: The team's name, such as Team 1.
        subtask: The subtask the team works on.
        members: The team's agents. The first one leads.
    """

    name: str
    subtask: str
    members: List[Any] = field(default_factory=list)

    @property
    def leader(self) -> Any:
        """The member that bid highest, who writes the team result."""
        return self.members[0]


def assemble_teams(
    bids: Dict[str, Tuple[Optional[int], float]],
    num_subtasks: int,
    max_team_size: Optional[int] = None,
) -> List[List[str]]:
    """Place each agent on the subtask it bid for, then fill empty ones.

    Agents are placed highest score first. One whose bid names no valid
    subtask, or whose subtask is full, joins the smallest team. A subtask
    nobody chose takes the lowest bidder from the largest team that can
    spare one.

    Args:
        bids: Each agent's chosen subtask index, or None, and its score.
        num_subtasks: How many subtasks there are.
        max_team_size: Most agents on one team. None for no limit.

    Returns:
        List[List[str]]: Agent names for each subtask, highest bidder first.
    """
    teams: List[List[str]] = [[] for _ in range(num_subtasks)]
    ranked = sorted(bids, key=lambda name: -bids[name][1])

    for name in ranked:
        choice = bids[name][0]
        valid = choice is not None and 0 <= choice < num_subtasks
        full = (
            valid
            and max_team_size is not None
            and len(teams[choice]) >= max_team_size
        )

        if not valid or full:
            choice = min(
                range(num_subtasks), key=lambda i: len(teams[i])
            )

        teams[choice].append(name)

    for team in teams:
        donor = max(teams, key=len)
        if not team and len(donor) > 1:
            team.append(donor.pop())

    return teams


def _tool_arguments(output: Any) -> Dict[str, Any]:
    """Return the arguments of the first tool call in a model reply.

    Args:
        output: A list of tool calls, one call, or the text of one.

    Returns:
        Dict[str, Any]: The parsed arguments, or an empty dict.
    """
    if isinstance(output, str):
        try:
            output = ast.literal_eval(output)
        except (ValueError, SyntaxError):
            return {}

    if isinstance(output, list):
        output = output[0] if output else None

    function = (
        output.get("function")
        if isinstance(output, dict)
        else getattr(output, "function", None)
    )
    arguments = (
        function.get("arguments")
        if isinstance(function, dict)
        else getattr(function, "arguments", None)
    )

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}

    return arguments if isinstance(arguments, dict) else {}


def _call_tool(
    model_name: str,
    system_prompt: Optional[str],
    prompt: str,
    tool: Dict[str, Any],
    usage_hook: Optional[Callable[[dict], None]] = None,
) -> Dict[str, Any]:
    """Make one model call that must answer with the given tool.

    Args:
        model_name: The model to call.
        system_prompt: The system prompt to send with the call.
        prompt: The user turn.
        tool: The tool the model must call.
        usage_hook: Receives the call's token usage.

    Returns:
        Dict[str, Any]: The tool call's arguments, or an empty dict.
    """
    llm = LiteLLM(
        model_name=model_name,
        system_prompt=system_prompt,
        tools_list_dictionary=[tool],
        tool_choice=forced_tool_choice(tool),
        usage_hook=usage_hook,
    )

    return _tool_arguments(llm.run(task=prompt))


def _numbered(items: List[str]) -> str:
    """Number each item on its own line, starting from 1."""
    return "\n".join(
        f"{number}. {item}" for number, item in enumerate(items, 1)
    )


class TeamFormationSwarm:
    """Agents bid on subtasks, form teams around them, and share results across teams.

    A coordinator splits the task into subtasks. Every agent bids for the
    subtask it suits best, and the agents that chose the same subtask form
    a team led by its highest bidder. Teams work at the same time. In each
    loop every member adds to its team's discussion, the leader writes the
    team result, and all teams read each other's results in the next loop.
    The coordinator then combines the results into one answer.

    Args:
        agents: The agents that assemble into teams. Names must be unique.
        name: Name of the swarm.
        description: What the swarm is for.
        max_loops: Rounds of team work. Each ends with every team sharing
            its result.
        max_team_size: Most agents one team takes. An agent whose subtask
            is full joins the smallest team. None for no limit.
        coordinator_model_name: Model that splits the task and combines
            the results.
        output_type: Format of the returned conversation. The default,
            final, returns the combined answer.
    """

    def __init__(
        self,
        agents: List[Agent],
        name: str = "TeamFormationSwarm",
        description: str = (
            "Agents bid on subtasks, form teams and share results "
            "across teams."
        ),
        max_loops: int = 1,
        max_team_size: Optional[int] = None,
        coordinator_model_name: str = "gpt-5.4",
        output_type: OutputType = "final",
    ):
        self.agents = agents
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.max_team_size = max_team_size
        self.coordinator_model_name = coordinator_model_name
        self.output_type = output_type
        self.reliability_check()

        self.coordinator = Agent(
            agent_name=f"{name}-Coordinator",
            agent_description="Splits the task and combines the results.",
            system_prompt=COORDINATOR_PROMPT,
            model_name=coordinator_model_name,
            max_loops=1,
            print_on=False,
        )
        self.conversation = Conversation(time_enabled=False)
        self.teams: List[Team] = []
        self.bids: Dict[str, Dict[str, Any]] = {}

        capture_init(self)

    def reliability_check(self) -> None:
        """Check the agents and settings before any run.

        Raises:
            ValueError: No agents, two agents share a name, max_loops is
                below 1, or max_team_size is below 1.
        """
        if not self.agents:
            raise ValueError(
                "TeamFormationSwarm needs at least one agent"
            )

        names = [agent.agent_name for agent in self.agents]
        if len(set(names)) != len(names):
            raise ValueError(
                "TeamFormationSwarm agents need unique names, since "
                "teams and bids are kept by name"
            )

        if not isinstance(self.max_loops, int) or self.max_loops < 1:
            raise ValueError(
                "max_loops must be an integer of at least 1"
            )

        if self.max_team_size is not None and self.max_team_size < 1:
            raise ValueError(
                "max_team_size must be at least 1 or None"
            )

    @trace_run(
        "TeamFormationSwarm.run",
        input_params=("task",),
        usage=lambda swarm: swarm.usage,
    )
    def run(self, task: str) -> Any:
        """Split the task, assemble teams, run them and combine their results.

        Args:
            task: The task for the whole swarm.

        Returns:
            Any: The conversation in output_type format.
        """
        reset_agents([*self.agents, self.coordinator])
        self.conversation = Conversation(time_enabled=False)
        self.conversation.add(role="User", content=task)

        subtasks = self._plan(task)
        self.teams = self._assemble(task, subtasks)

        results: Dict[str, str] = {}
        for _ in range(self.max_loops):
            results = self._run_teams(task, results)

        prompt = SYNTHESIS_PROMPT.format(
            task=task, results=self._board(results)
        )
        answer = agent_answer(
            self.coordinator,
            fallback=self.coordinator.run(task=prompt),
        )
        self.conversation.add(
            role=self.coordinator.agent_name, content=answer
        )

        return history_output_formatter(
            self.conversation, self.output_type
        )

    def _plan(self, task: str) -> List[str]:
        """Ask the coordinator to split the task into subtasks.

        Args:
            task: The task for the whole swarm.

        Returns:
            List[str]: At most one subtask per agent, or the task itself
                when the coordinator returns none.
        """
        prompt = PLAN_PROMPT.format(
            task=task, max_subtasks=len(self.agents)
        )
        arguments = _call_tool(
            self.coordinator.model_name,
            COORDINATOR_PROMPT,
            prompt,
            PLAN_TOOL,
            self.coordinator._add_usage,
        )

        subtasks = [
            str(subtask).strip()
            for subtask in arguments.get("subtasks") or []
            if str(subtask).strip()
        ]
        subtasks = subtasks[: len(self.agents)] or [task]

        self.conversation.add(
            role=self.coordinator.agent_name,
            content="Subtasks:\n" + _numbered(subtasks),
        )
        return subtasks

    def _assemble(self, task: str, subtasks: List[str]) -> List[Team]:
        """Collect every agent's bid and form one team per subtask.

        Args:
            task: The task for the whole swarm.
            subtasks: The subtasks from the coordinator.

        Returns:
            List[Team]: One team per subtask.
        """
        prompt = BID_PROMPT.format(
            task=task, subtasks=_numbered(subtasks)
        )

        with ContextThreadPoolExecutor(
            max_workers=len(self.agents)
        ) as pool:
            bids = list(
                pool.map(
                    lambda agent: self._bid(agent, prompt),
                    self.agents,
                )
            )

        self.bids = {
            agent.agent_name: bid
            for agent, bid in zip(self.agents, bids)
        }
        groups = assemble_teams(
            {
                name: (bid["subtask"], bid["score"])
                for name, bid in self.bids.items()
            },
            len(subtasks),
            self.max_team_size,
        )

        by_name = {agent.agent_name: agent for agent in self.agents}
        teams = [
            Team(
                name=f"Team {number}",
                subtask=subtask,
                members=[by_name[name] for name in group],
            )
            for number, (subtask, group) in enumerate(
                zip(subtasks, groups), 1
            )
        ]

        self.conversation.add(
            role=self.name,
            content="\n".join(
                f"{team.name} ({team.subtask}): "
                + ", ".join(m.agent_name for m in team.members)
                for team in teams
            ),
        )
        return teams

    def _bid(self, agent: Agent, prompt: str) -> Dict[str, Any]:
        """Ask one agent which subtask it wants and how strongly.

        The bid is a separate call with the agent's model and system
        prompt, so it leaves the agent's memory and tools untouched.

        Args:
            agent: The agent bidding.
            prompt: The task and the numbered subtasks.

        Returns:
            Dict[str, Any]: The chosen subtask index or None, the score
                from 0 to 1, and the reason.
        """
        try:
            arguments = _call_tool(
                agent.model_name,
                agent.system_prompt,
                prompt,
                BID_TOOL,
                getattr(agent, "_add_usage", None),
            )
        except Exception as error:
            logger.warning(
                f"[{self.name}] {agent.agent_name} could not bid: {error}"
            )
            arguments = {}

        try:
            subtask = int(arguments["subtask"]) - 1
        except (KeyError, TypeError, ValueError):
            subtask = None

        try:
            score = min(1.0, max(0.0, float(arguments["score"])))
        except (KeyError, TypeError, ValueError):
            score = 0.0

        return {
            "subtask": subtask,
            "score": score,
            "reason": str(arguments.get("reason", "")),
        }

    def _run_teams(
        self, task: str, results: Dict[str, str]
    ) -> Dict[str, str]:
        """Run one loop of every team at once and collect their results.

        Args:
            task: The task for the whole swarm.
            results: Each team's result from the loop before, by name.

        Returns:
            Dict[str, str]: Each team's new result, by team name.
        """
        board = self._board(results)

        with ContextThreadPoolExecutor(
            max_workers=len(self.teams)
        ) as pool:
            discussions = list(
                pool.map(
                    lambda team: self._team_turn(task, team, board),
                    self.teams,
                )
            )

        for discussion in discussions:
            for speaker, reply in discussion:
                self.conversation.add(role=speaker, content=reply)

        return {
            team.name: discussion[-1][1]
            for team, discussion in zip(self.teams, discussions)
        }

    def _team_turn(
        self, task: str, team: Team, board: str
    ) -> List[Tuple[str, str]]:
        """Run one team's members in turn, leader last.

        Args:
            task: The task for the whole swarm.
            team: The team taking its turn.
            board: The results all teams shared in the loop before.

        Returns:
            List[Tuple[str, str]]: Each speaker's name and reply. The last
                one is the leader's team result.
        """
        discussion: List[Tuple[str, str]] = []

        for member in [*team.members[1:], team.leader]:
            teammates = ", ".join(
                other.agent_name
                for other in team.members
                if other is not member
            )
            prompt = TEAM_TURN_PROMPT.format(
                task=task,
                team=team.name,
                teammates=teammates or "no teammates",
                subtask=team.subtask,
                board=board,
                discussion="\n\n".join(
                    f"{speaker}: {reply}"
                    for speaker, reply in discussion
                )
                or "(you speak first)",
                instruction=(
                    LEADER_INSTRUCTION
                    if member is team.leader
                    else MEMBER_INSTRUCTION
                ),
            )

            try:
                output = member.run(task=prompt)
            except Exception as error:
                logger.warning(
                    f"[{self.name}] {member.agent_name} failed: {error}"
                )
                output = error

            reply = agent_answer(member, fallback=output)
            discussion.append((member.agent_name, str(reply)))

        return discussion

    def _board(self, results: Dict[str, str]) -> str:
        """Lay out each team's latest result under its subtask.

        Args:
            results: Each team's result, by team name.

        Returns:
            str: One block per team, or a note that nothing is shared yet.
        """
        if not results:
            return "(nothing shared yet)"

        return "\n\n".join(
            f"{team.name} ({team.subtask}):\n{results[team.name]}"
            for team in self.teams
            if team.name in results
        )

    @property
    def usage(self) -> dict:
        """Provider token usage summed over every agent, bids included.

        Returns:
            dict: input_tokens, output_tokens, cached_tokens,
                reasoning_tokens and total_tokens across every run.
        """
        return total_usage(self)
