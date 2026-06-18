"""
AuctionSwarm
================

Tasks are announced to a pool of agents; each agent bids on the task via a
forced ``bid(confidence, estimated_cost)`` tool call, and an auctioneer awards
the task to the agent(s) with the best score.

This avoids relying on a boss LLM's (possibly stale or wrong) model of which
agent is best for a task. Instead, each agent self-assesses its own confidence
and cost for the specific task at hand, and the auctioneer picks winners from
those bids using a pluggable scoring function.

Flow:

    task -> broadcast to every agent in the pool
    -> each agent calls bid(confidence, estimated_cost) via a forced function
       call, scored by `scoring` (default: confidence per unit cost)
    -> the top_k highest-scoring agents execute the task
    -> the auctioneer keeps the best of their results

Key concepts:

    BID_TOOL
        A forced function-calling schema used to make every agent return a
        structured ``(confidence, estimated_cost)`` bid instead of free-form
        text.

    top_k
        How many of the highest-scoring bidders actually execute the task.

    scoring
        Either the name of a built-in scoring function (currently
        ``"confidence_per_cost"``) or a callable
        ``(confidence, estimated_cost) -> float`` used to rank bids.

Example:

    from swarms import Agent, AuctionSwarm

    agents = [
        Agent(agent_name="Translator", model_name="gpt-5.4"),
        Agent(agent_name="Generalist", model_name="gpt-5.4"),
    ]

    swarm = AuctionSwarm(agents=agents, top_k=1)
    result = swarm.run("Translate this contract into plain English.")
"""

import ast
import json
from typing import Any, Callable, List, Optional, Tuple, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.serialization import SerializableMixin
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import OutputType

BID_TOOL = {
    # The LLM is forced to call this function so every bid is a structured,
    # machine-readable (confidence, estimated_cost) pair.
    "type": "function",
    "function": {
        "name": "bid",
        "description": (
            "Submit a bid for this task. Report how confident you are that you "
            "can complete it well, and your estimated relative cost to do so."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "description": "How confident you are (0 = no chance, 1 = certain) that you can complete this task well.",
                    "minimum": 0,
                    "maximum": 1,
                },
                "estimated_cost": {
                    "type": "number",
                    "description": "Your estimated relative cost to complete this task. Use 1.0 as an average cost; lower means cheaper/faster, higher means more expensive/slower.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["confidence", "estimated_cost"],
        },
    },
}

# Lower bound used when clamping/dividing by estimated_cost so a
# zero-cost bid can never produce a divide-by-zero or an infinite score.
MIN_ESTIMATED_COST = 1e-6

# Tool-call output above this length is rejected by _extract_bid without
# attempting ast.literal_eval, as a guard against pathological model
# output causing excessive parse time.
MAX_TOOL_OUTPUT_LEN = 10_000

BID_PROMPT = """You are {agent_name}, one of several agents bidding to handle a task.

Task:
{task}

Decide how confident you are that you can complete this task well, and
estimate your relative cost (time, effort, tokens) to do so.

Call the `bid` function with:
  - confidence: 0..1, how likely you are to produce a high-quality result.
  - estimated_cost: a positive number representing your relative cost. Use
    1.0 as an average cost; lower than 1 means cheaper/faster, higher means
    more expensive/slower.

Be honest — overbidding on tasks outside your expertise wastes the auction.
"""


def _extract_bid(tool_output: Any) -> Tuple[float, float]:
    """Parse and normalize a forced ``bid()`` tool call.

    Args:
        tool_output: Raw provider output from the forced tool invocation. Some
            providers return a single tool-call dictionary, while others
            return a list of tool-call dictionaries.

    Returns:
        A ``(confidence, estimated_cost)`` pair. Invalid or missing output is
        treated as a no-confidence bid: ``(0.0, 1.0)``. Confidence is clamped
        into the ``0..1`` range and cost is clamped to be strictly positive.
    """
    if isinstance(tool_output, str):
        # Agents with output_type="str-all-except-first" (the default)
        # return tool calls as the str() of a list of dicts, not JSON.
        if len(tool_output) > MAX_TOOL_OUTPUT_LEN:
            return 0.0, 1.0
        try:
            tool_output = ast.literal_eval(tool_output)
        except (ValueError, SyntaxError):
            return 0.0, 1.0

    if isinstance(tool_output, list):
        tool_output = tool_output[0] if tool_output else None
    if not tool_output:
        return 0.0, 1.0

    fn = (
        tool_output.get("function")
        if isinstance(tool_output, dict)
        else None
    )
    if not fn:
        return 0.0, 1.0

    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return 0.0, 1.0
    if not isinstance(args, dict):
        return 0.0, 1.0

    try:
        confidence = float(args.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        estimated_cost = float(args.get("estimated_cost", 1.0))
    except (TypeError, ValueError):
        estimated_cost = 1.0

    confidence = max(0.0, min(1.0, confidence))
    estimated_cost = max(MIN_ESTIMATED_COST, estimated_cost)
    return confidence, estimated_cost


def confidence_per_cost(
    confidence: float, estimated_cost: float
) -> float:
    """Default scoring function: confidence divided by estimated cost."""
    return confidence / max(estimated_cost, MIN_ESTIMATED_COST)


SCORING_FUNCTIONS = {
    "confidence_per_cost": confidence_per_cost,
}


class _BidWrapper:
    """Wraps an agent so ``run_agents_concurrently`` collects a bid for it.

    ``run_agents_concurrently`` calls ``agent.run(task=...)`` on whatever it
    is given. This wrapper substitutes the bidding prompt (with this agent's
    name filled in) for whatever task it is called with.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.agent_name = agent.agent_name

    def run(self, task: str) -> Any:
        prompt = BID_PROMPT.format(
            agent_name=self.agent.agent_name, task=task
        )
        return self.agent.run(task=prompt)


class AuctionSwarm(SerializableMixin):
    """Award a task to the agent(s) whose self-assessed bid scores best.

    ``AuctionSwarm`` broadcasts a task to every agent in the pool, collects a
    structured ``(confidence, estimated_cost)`` bid from each via a forced
    ``bid`` tool call, scores the bids, and runs the ``top_k`` winners on the
    task. When ``top_k`` is 1, the single winner's response is the result.
    When ``top_k`` is greater than 1, every winner runs and the response from
    the highest-scoring winner that did not error is kept.
    """

    _to_dict_exclude = ("agents", "conversation")

    def __init__(
        self,
        name: str = "auction-swarm",
        description: str = "Agents bid for tasks; the best bid wins.",
        agents: Optional[List[Agent]] = None,
        top_k: int = 1,
        scoring: Union[str, Callable[[float, float], float]] = (
            "confidence_per_cost"
        ),
        output_type: OutputType = "dict",
        verbose: bool = False,
        print_on: bool = True,
        auto_equip: bool = True,
    ):
        """Initialize the auction.

        Args:
            name: Human-readable name used in logs and serialized state.
            description: Short description of the swarm structure.
            agents: The bidding pool. At least one agent is required.
            top_k: Number of highest-scoring bidders that execute the task.
            scoring: Either the name of a built-in scoring function
                (``"confidence_per_cost"``) or a callable
                ``(confidence, estimated_cost) -> float`` used to rank bids.
            output_type: Format passed to ``history_output_formatter``.
            verbose: Whether to emit internal log messages.
            print_on: Whether to print the auction results.
            auto_equip: Whether to inject ``BID_TOOL`` into agents that don't
                already carry it.

        Raises:
            ValueError: If no agents are provided, ``top_k`` is less than 1,
                or ``scoring`` names an unknown built-in function.
        """
        self.name = name
        self.description = description
        self.agents = agents or []
        self.top_k = top_k
        self.output_type = output_type
        self.verbose = verbose
        self.print_on = print_on
        self.auto_equip = auto_equip

        if not self.agents:
            raise ValueError(
                "AuctionSwarm requires at least 1 agent."
            )
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if isinstance(scoring, str):
            try:
                self.scoring_fn = SCORING_FUNCTIONS[scoring]
            except KeyError:
                raise ValueError(
                    f"Unknown scoring function '{scoring}'. "
                    f"Available: {list(SCORING_FUNCTIONS)}"
                )
        else:
            self.scoring_fn = scoring

        self.conversation = Conversation()

    def _ensure_bid_tool(self) -> None:
        """Inject ``BID_TOOL`` into agents that do not already carry it.

        Without the ``bid`` tool an agent's bid can never be parsed, so it
        would always lose the auction with a no-confidence bid. The agent's
        LLM client bakes ``tools_list_dictionary`` in at construction time, so
        after appending the schema the client is rebuilt via
        ``llm_handling()``.
        """
        for agent in self.agents:
            tools = agent.tools_list_dictionary or []
            if any(
                tool.get("function", {}).get("name") == "bid"
                for tool in tools
                if isinstance(tool, dict)
            ):
                continue
            agent.tools_list_dictionary = [*tools, BID_TOOL]
            agent.llm = agent.llm_handling()
            self._log(
                "info", f"Injected bid tool into {agent.agent_name}"
            )

    def _remove_bid_tool(self) -> None:
        """Strip ``BID_TOOL`` from every agent that carries it.

        Leaving ``bid`` in an agent's ``tools_list_dictionary`` causes the
        winning agent's real task execution to call ``bid`` again instead of
        producing its actual output, so the tool is removed from every agent
        before the winners run (it is re-added for the next auction by
        ``_ensure_bid_tool``) and the LLM client is rebuilt via
        ``llm_handling()``.
        """
        for agent in self.agents:
            tools = agent.tools_list_dictionary or []
            if not any(
                isinstance(tool, dict)
                and tool.get("function", {}).get("name") == "bid"
                for tool in tools
            ):
                continue
            agent.tools_list_dictionary = [
                tool
                for tool in tools
                if not (
                    isinstance(tool, dict)
                    and tool.get("function", {}).get("name") == "bid"
                )
            ]
            agent.llm = agent.llm_handling()
            self._log(
                "info", f"Removed bid tool from {agent.agent_name}"
            )

    def _run_auction(
        self, task: str
    ) -> List[Tuple[Agent, float, float, float]]:
        """Collect and score bids from every agent.

        Returns:
            A list of ``(agent, confidence, estimated_cost, score)`` tuples
            sorted by score, highest first.
        """
        tool_outputs = run_agents_concurrently(
            [_BidWrapper(agent) for agent in self.agents],
            task=task,
            return_agent_output_dict=True,
        )

        bids = []
        for agent in self.agents:
            tool_output = tool_outputs.get(agent.agent_name)
            if isinstance(tool_output, Exception):
                self._log(
                    "warning",
                    f"{agent.agent_name} failed to bid: {tool_output}",
                )
                confidence, estimated_cost = 0.0, 1.0
            else:
                confidence, estimated_cost = _extract_bid(tool_output)

            score = self.scoring_fn(confidence, estimated_cost)
            bids.append((agent, confidence, estimated_cost, score))

        bids.sort(key=lambda b: b[3], reverse=True)
        return bids

    def run(self, task: str) -> Any:
        """Run one auction round and execute the winning agent(s).

        Args:
            task: The task to auction off and execute.

        Returns:
            Formatted conversation output from ``history_output_formatter``.
        """
        self._log("info", f"[{self.name}] auctioning task: {task}")

        self.conversation.add(role="User", content=task)

        if self.auto_equip:
            self._ensure_bid_tool()
        try:
            bids = self._run_auction(task)
        finally:
            if self.auto_equip:
                self._remove_bid_tool()

        # Every agent's run() call during bidding left the bid prompt and
        # the forced bid tool call in its short_memory. With
        # output_type="str-all-except-first" (the default), that would leak
        # into a later execution run's output as extra text ahead of the
        # real response, so reset short_memory for every bidder now.
        for agent in self.agents:
            agent.short_memory = agent.short_memory_init()

        if self.print_on:
            bid_summary = "\n".join(
                f"{agent.agent_name}: confidence={confidence:.2f}, "
                f"estimated_cost={estimated_cost:.2f}, score={score:.4f}"
                for agent, confidence, estimated_cost, score in bids
            )
            formatter.print_panel(
                bid_summary, title=f"{self.name} - Bids"
            )

        self.conversation.add(
            role=self.name,
            content=(
                "Bids: "
                + ", ".join(
                    f"{agent.agent_name}={score:.4f}"
                    for agent, _, _, score in bids
                )
            ),
        )

        winners = bids[: self.top_k]

        results = run_agents_concurrently(
            [agent for agent, *_ in winners],
            task=task,
            return_agent_output_dict=True,
        )

        # `winners` is sorted by score, highest first, so the first entry
        # that did not error is the auctioneer's pick for "best result".
        successful = []
        for agent, *_ in winners:
            response = results.get(agent.agent_name)
            if isinstance(response, Exception):
                self._log(
                    "warning",
                    f"{agent.agent_name} failed to execute: {response}",
                )
                continue
            successful.append((agent, response))

        for agent, response in successful[1:]:
            self.conversation.add(
                role=agent.agent_name, content=response
            )

        if successful:
            best_agent, best_response = successful[0]
            self.conversation.add(
                role=best_agent.agent_name, content=best_response
            )
            if self.print_on:
                formatter.print_panel(
                    f"Awarded to {best_agent.agent_name}",
                    title=f"{self.name} - Award",
                )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def __call__(self, task: str) -> Any:
        """Convenience alias for :meth:`run` so the swarm can be invoked
        directly: ``swarm(task)``.
        """
        return self.run(task)

    def batch_run(self, tasks: List[str]) -> List[Any]:
        """Run an auction for each task in sequence.

        Args:
            tasks: List of tasks to auction off and execute.

        Returns:
            List of formatted conversation outputs from :meth:`run`.
        """
        return [self.run(task) for task in tasks]
