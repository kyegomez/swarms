import json
import math
from typing import Any, Dict, List, Optional

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.swarm_id import swarm_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM
from swarms.utils.output_types import OutputType

TOURNAMENT_JUDGE_PROMPT = """
You are an impartial pairwise judge in a tournament of candidate answers.

You will be shown the original task and exactly two candidate answers, labeled
Candidate A and Candidate B. Your only job is to decide which of the two answers
is the stronger response to the task.

Evaluation Criteria:
1. Correctness and factual accuracy
2. Completeness — how fully the answer addresses the task
3. Clarity, structure, and communication quality
4. Relevance — staying on task without padding or digression

Judging Guidelines:
- Compare the two answers directly against each other; do not score them in isolation.
- Ignore answer ordering, length for its own sake, and stylistic flourishes that
  do not improve the response.
- Remain impartial: judge only the content, never the identity of the candidate.
- You must always pick exactly one winner, even when the answers are close.

You must respond by calling the pick_winner function with your choice and a
concise justification.
"""

PICK_WINNER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "pick_winner",
            "description": "Pick the stronger of the two candidate answers to the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "winner": {
                        "type": "string",
                        "enum": ["a", "b"],
                        "description": "The winning candidate: 'a' for Candidate A, 'b' for Candidate B.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "A concise justification for why the chosen answer is stronger.",
                    },
                },
                "required": ["winner", "reasoning"],
            },
        },
    }
]

BRACKET_TYPES = ["single-elimination", "swiss"]


class TournamentSwarm:
    """
    A tournament-style answer selection system built on pairwise judging.

    N candidate agents answer the task independently and concurrently. The
    answers are then seeded into a bracket and eliminated through head-to-head
    matches: each match presents exactly two answers to a judge, which picks a
    winner via a forced `pick_winner` function call. The final survivor's
    answer is the result.

    This fixes the weakness of judges at absolute scoring of long-form
    answers — every judgment is a two-way comparison, a far easier task for a
    judge model, at O(N) comparisons instead of asking one judge to rank N
    answers in absolute terms.

    Bracket types:
        - "single-elimination" (default): losers are knocked out each round
          until one candidate remains. Top seeds receive byes on odd rounds.
        - "swiss": no eliminations; candidates are paired against opponents
          with equal scores for ceil(log2(N)) rounds (configurable), and the
          highest scorer wins. Ties are broken by head-to-head result, then
          by seeding (input order).

    Attributes:
        agents (List[Agent]): The candidate generator agents.
        judge (Optional[Agent]): Optional judge agent whose system prompt and
            model configure the pairwise comparator.
        bracket (str): The bracket type, "single-elimination" or "swiss".
        bracket_history (List[dict]): Per-round match records from the last run.
        winner (Optional[dict]): The winning candidate from the last run, as a
            dict with "name" and "answer" keys.

    Examples:
        >>> swarm = TournamentSwarm(
        ...     agents=[writer_a, writer_b, writer_c, writer_d],
        ...     judge=Agent(agent_name="Judge", model_name="gpt-4.1", max_loops=1),
        ...     bracket="single-elimination",
        ... )
        >>> result = swarm.run("Write the strongest possible launch announcement.")
    """

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "TournamentSwarm",
        description: str = "Runs candidate agents in a bracket of pairwise judge comparisons until one answer survives",
        agents: List[Agent] = None,
        judge: Optional[Agent] = None,
        bracket: str = "single-elimination",
        judge_model_name: str = "gpt-4.1",
        judge_system_prompt: str = TOURNAMENT_JUDGE_PROMPT,
        swiss_rounds: Optional[int] = None,
        output_type: OutputType = "final",
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the TournamentSwarm.

        Args:
            id (str): Unique identifier for the swarm.
            name (str): Name of the swarm.
            description (str): Description of the swarm's purpose.
            agents (List[Agent]): Candidate agents that each answer the task.
                At least 2 agents with unique names are required.
            judge (Optional[Agent]): Optional judge agent. Its system prompt,
                model name, and temperature configure the pairwise comparator.
                If omitted, a default judge is built from judge_model_name and
                judge_system_prompt.
            bracket (str): Bracket type — "single-elimination" or "swiss".
            judge_model_name (str): Model for the default judge when no judge
                agent is provided.
            judge_system_prompt (str): System prompt for the default judge.
            swiss_rounds (Optional[int]): Number of Swiss rounds. Defaults to
                ceil(log2(N)) when bracket is "swiss".
            output_type (OutputType): Output format for the result. The default
                "final" returns the winning answer; conversation-history formats
                ("dict", "str", ...) include all candidate answers and verdicts.
            verbose (bool): Whether to enable verbose logging.

        Raises:
            ValueError: If agents are missing, fewer than 2, have duplicate
                names, or if bracket is not a supported type.
        """
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.judge = judge
        self.bracket = bracket
        self.swiss_rounds = swiss_rounds
        self.output_type = output_type
        self.verbose = verbose

        self.bracket_history: List[dict] = []
        self.winner: Optional[dict] = None

        self.reliability_check()

        if judge is not None:
            self.judge_name = judge.agent_name
            judge_prompt = judge.system_prompt or judge_system_prompt
            judge_model = judge.model_name
            judge_temperature = judge.temperature
        else:
            self.judge_name = "Tournament-Judge"
            judge_prompt = judge_system_prompt
            judge_model = judge_model_name
            judge_temperature = 0.0

        # Pairwise comparator with a forced pick_winner function call,
        # following the function-caller pattern used by MultiAgentRouter
        # and HeavySwarm.
        self.judge_caller = LiteLLM(
            model_name=judge_model,
            system_prompt=judge_prompt,
            temperature=judge_temperature,
            tools_list_dictionary=PICK_WINNER_TOOL,
            tool_choice={
                "type": "function",
                "function": {"name": "pick_winner"},
            },
            *args,
            **kwargs,
        )

        self.conversation = Conversation(time_enabled=False)

    def reliability_check(self):
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list is empty")

        if len(self.agents) < 2:
            raise ValueError(
                "TournamentSwarm requires at least 2 candidate agents"
            )

        names = [agent.agent_name for agent in self.agents]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Agent names must be unique, got: {names}"
            )

        if self.bracket not in BRACKET_TYPES:
            raise ValueError(
                f"Invalid bracket type: '{self.bracket}'. "
                f"Available types: {', '.join(BRACKET_TYPES)}"
            )

        if self.swiss_rounds is not None and self.swiss_rounds < 1:
            raise ValueError("swiss_rounds must be at least 1")

    def run(self, task: str, *args, **kwargs) -> Any:
        """
        Run the tournament on a task and return the surviving answer.

        Args:
            task (str): The task for the candidate agents to answer.
            *args: Additional positional arguments passed to the agents.
            **kwargs: Additional keyword arguments passed to the agents.

        Returns:
            Any: The winning answer (output_type="final") or the formatted
                conversation history, depending on output_type.

        Raises:
            ValueError: If task is empty.
            RuntimeError: If every candidate agent fails to produce an answer.
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        self.conversation = Conversation(time_enabled=False)
        self.bracket_history = []
        self.winner = None

        self.conversation.add(role="user", content=task)

        if self.verbose:
            logger.info(
                f"TournamentSwarm: running {len(self.agents)} candidates "
                f"on a {self.bracket} bracket"
            )

        outputs = run_agents_concurrently(
            agents=self.agents,
            task=task,
            return_agent_output_dict=True,
        )

        candidates = []
        for agent in self.agents:
            answer = outputs.get(agent.agent_name)
            if answer is None or isinstance(answer, Exception):
                logger.warning(
                    f"TournamentSwarm: candidate '{agent.agent_name}' failed "
                    f"and is excluded from the bracket: {answer}"
                )
                continue
            self.conversation.add(
                role=agent.agent_name, content=answer
            )
            candidates.append(
                {"name": agent.agent_name, "answer": answer}
            )

        if len(candidates) == 0:
            raise RuntimeError(
                "TournamentSwarm: all candidate agents failed to produce an answer"
            )

        if len(candidates) == 1:
            winner = candidates[0]
        elif self.bracket == "swiss":
            winner = self._run_swiss(task, candidates)
        else:
            winner = self._run_single_elimination(task, candidates)

        self.winner = winner
        self.conversation.add(
            role=f"Tournament-Winner ({winner['name']})",
            content=winner["answer"],
        )

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def batch_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Run the tournament sequentially on a batch of tasks.

        Args:
            tasks (List[str]): List of tasks to run tournaments for.

        Returns:
            List[Any]: The result of each tournament, one per task.
        """
        return [self.run(task, *args, **kwargs) for task in tasks]

    def get_bracket(self) -> dict:
        """
        Get the full bracket of the last run as metadata.

        Returns:
            dict: The bracket type, all per-round match records, and the
                winning candidate's name.
        """
        return {
            "bracket": self.bracket,
            "rounds": self.bracket_history,
            "winner": self.winner["name"] if self.winner else None,
        }

    def _run_single_elimination(
        self, task: str, candidates: List[dict]
    ) -> dict:
        """
        Run a single-elimination bracket until one candidate survives.

        Candidates are seeded in input order. On rounds with an odd number of
        survivors, the top remaining seed receives a bye.

        Args:
            task (str): The original task, shown to the judge in each match.
            candidates (List[dict]): Seeded candidates with "name" and "answer".

        Returns:
            dict: The winning candidate.
        """
        round_num = 0
        while len(candidates) > 1:
            round_num += 1
            matches = []
            pool = list(candidates)
            next_round = []

            if len(pool) % 2 == 1:
                bye = pool.pop(0)
                next_round.append(bye)
                matches.append({"bye": bye["name"]})
                if self.verbose:
                    logger.info(
                        f"TournamentSwarm round {round_num}: bye for '{bye['name']}'"
                    )

            for i in range(0, len(pool), 2):
                candidate_a, candidate_b = pool[i], pool[i + 1]
                verdict = self._run_match(
                    task, candidate_a, candidate_b, round_num
                )
                matches.append(verdict)
                next_round.append(
                    candidate_a
                    if verdict["winner"] == candidate_a["name"]
                    else candidate_b
                )

            self.bracket_history.append(
                {"round": round_num, "matches": matches}
            )
            candidates = next_round

        return candidates[0]

    def _run_swiss(
        self, task: str, candidates: List[dict]
    ) -> dict:
        """
        Run a Swiss bracket: no eliminations, candidates are paired against
        equal scorers each round, and the highest scorer wins.

        Args:
            task (str): The original task, shown to the judge in each match.
            candidates (List[dict]): Seeded candidates with "name" and "answer".

        Returns:
            dict: The winning candidate.
        """
        rounds = self.swiss_rounds or max(
            1, math.ceil(math.log2(len(candidates)))
        )
        scores = {candidate["name"]: 0 for candidate in candidates}
        seeds = {
            candidate["name"]: i
            for i, candidate in enumerate(candidates)
        }
        played = set()
        head_to_head = {}

        for round_num in range(1, rounds + 1):
            standings = sorted(
                candidates,
                key=lambda c: (-scores[c["name"]], seeds[c["name"]]),
            )
            matches = []

            if len(standings) % 2 == 1:
                bye = standings.pop()
                scores[bye["name"]] += 1
                matches.append({"bye": bye["name"]})
                if self.verbose:
                    logger.info(
                        f"TournamentSwarm Swiss round {round_num}: bye for '{bye['name']}'"
                    )

            unpaired = list(standings)
            while unpaired:
                candidate_a = unpaired.pop(0)
                # Prefer the highest-ranked opponent not yet faced;
                # fall back to a rematch if every opponent was faced.
                opponent_index = 0
                for j, opponent in enumerate(unpaired):
                    pair = frozenset(
                        (candidate_a["name"], opponent["name"])
                    )
                    if pair not in played:
                        opponent_index = j
                        break
                candidate_b = unpaired.pop(opponent_index)
                played.add(
                    frozenset(
                        (candidate_a["name"], candidate_b["name"])
                    )
                )

                verdict = self._run_match(
                    task, candidate_a, candidate_b, round_num
                )
                matches.append(verdict)
                scores[verdict["winner"]] += 1
                head_to_head[
                    frozenset(
                        (candidate_a["name"], candidate_b["name"])
                    )
                ] = verdict["winner"]

            self.bracket_history.append(
                {
                    "round": round_num,
                    "matches": matches,
                    "scores": dict(scores),
                }
            )

        top_score = max(scores.values())
        leaders = sorted(
            (c for c in candidates if scores[c["name"]] == top_score),
            key=lambda c: seeds[c["name"]],
        )

        # Tie-break: head-to-head between two leaders if they met,
        # otherwise seeding (input order).
        if len(leaders) == 2:
            pair = frozenset(
                (leaders[0]["name"], leaders[1]["name"])
            )
            if pair in head_to_head:
                winner_name = head_to_head[pair]
                return next(
                    c for c in leaders if c["name"] == winner_name
                )

        return leaders[0]

    def _run_match(
        self,
        task: str,
        candidate_a: dict,
        candidate_b: dict,
        round_num: int,
    ) -> dict:
        """
        Run a single head-to-head match between two candidate answers.

        The judge receives the task and both answers and must pick a winner
        through a forced pick_winner function call.

        Args:
            task (str): The original task.
            candidate_a (dict): First candidate with "name" and "answer".
            candidate_b (dict): Second candidate with "name" and "answer".
            round_num (int): The current round number (1-indexed).

        Returns:
            dict: Match record with round, both candidate names, the winning
                candidate's name, and the judge's reasoning.
        """
        match_prompt = (
            f"Original task:\n{task}\n\n"
            f"Candidate A:\n{candidate_a['answer']}\n\n"
            f"Candidate B:\n{candidate_b['answer']}\n\n"
            f"Compare the two candidate answers and call pick_winner with "
            f"your choice and reasoning."
        )

        choice, reasoning = "a", ""
        try:
            raw_output = self.judge_caller.run(match_prompt)
            choice, reasoning = self._parse_pick_winner(raw_output)
        except Exception as e:
            logger.warning(
                f"TournamentSwarm: judge call failed in round {round_num} "
                f"('{candidate_a['name']}' vs '{candidate_b['name']}'), "
                f"higher seed advances: {e}"
            )
            reasoning = f"Judge call failed ({e}); higher seed advances by default."

        winner = candidate_a if choice == "a" else candidate_b
        verdict = {
            "round": round_num,
            "candidate_a": candidate_a["name"],
            "candidate_b": candidate_b["name"],
            "winner": winner["name"],
            "reasoning": reasoning,
        }

        self.conversation.add(
            role=self.judge_name,
            content=(
                f"Round {round_num}: {candidate_a['name']} vs "
                f"{candidate_b['name']} — winner: {winner['name']}. "
                f"Reasoning: {reasoning}"
            ),
        )

        if self.verbose:
            logger.info(
                f"TournamentSwarm round {round_num}: '{winner['name']}' beats "
                f"'{candidate_a['name'] if winner is candidate_b else candidate_b['name']}'"
            )

        return verdict

    def _parse_pick_winner(self, raw_output: Any) -> tuple:
        """
        Parse the judge's pick_winner tool call into a (choice, reasoning) pair.

        Handles both object-style tool calls (ChatCompletionMessageToolCall)
        and dict-style tool calls returned by the LiteLLM wrapper.

        Args:
            raw_output (Any): The raw tool-call output from the judge call.

        Returns:
            tuple: (choice, reasoning) where choice is "a" or "b". Defaults to
                "a" if the output cannot be parsed.
        """
        tool_calls = raw_output
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not tool_calls:
            raise ValueError(
                f"Judge returned no tool calls: {raw_output}"
            )

        tool_call = tool_calls[0]
        if isinstance(tool_call, dict):
            arguments = tool_call.get("function", {}).get(
                "arguments", "{}"
            )
        else:
            arguments = tool_call.function.arguments

        data = (
            json.loads(arguments)
            if isinstance(arguments, str)
            else dict(arguments)
        )

        choice = str(data.get("winner", "a")).strip().lower()
        if choice not in ("a", "b"):
            logger.warning(
                f"TournamentSwarm: invalid winner choice '{choice}', defaulting to 'a'"
            )
            choice = "a"

        reasoning = str(data.get("reasoning", ""))
        return choice, reasoning
