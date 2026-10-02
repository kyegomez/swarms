"""
Tree of Thoughts reasoning agent.

Implements "Tree of Thoughts: Deliberate Problem Solving with Large Language
Models" (Yao et al., 2023, https://arxiv.org/abs/2305.10601). Instead of
answering in one pass, the agent grows a tree of partial solutions:

    1. Generate: from a node, propose ``num_thoughts`` candidate next steps,
       all in one call ("propose") or one call per step ("sample").
    2. Evaluate: score each candidate from 0 to 1, either on its own
       ("value") or by comparing candidates and voting ("vote").
    3. Search: explore breadth-first with a beam ("bfs") or depth-first with
       backtracking ("dfs"), pruning candidates below ``value_threshold``.
    4. Answer: write the final answer from the best path found.

Every model output is a function call validated against a Pydantic schema
(``ProposeThoughts``, ``EvaluateThought``, ``VoteForThought``,
``FinalAnswer``), so the search never parses free-form prose.

The prompts are domain-agnostic. To adapt the search to a task, describe what
one step looks like with ``thought_description`` and how to judge progress
with ``evaluation_criteria``.

Example:
    >>> from swarms.agents import TreeOfThoughts
    >>> agent = TreeOfThoughts(model_name="gpt-5.4", max_depth=3)
    >>> print(agent.run("Use 4, 9, 10 and 13 with + - * / to make 24."))
"""

import ast
import json
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    get_args,
)

from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)

from swarms.prompts.tree_of_thoughts_prompts import (
    TOT_EVALUATION_CRITERIA,
    TOT_FINAL_MARKER,
    TOT_FINAL_PROMPT,
    TOT_FINAL_STATUS_EMPTY,
    TOT_FINAL_STATUS_PARTIAL,
    TOT_FINAL_STATUS_SOLVED,
    TOT_GENERATE_PROMPT,
    TOT_LAST_STEP_RULE,
    TOT_MANY_CANDIDATES,
    TOT_NO_STEPS,
    TOT_ONE_CANDIDATE,
    TOT_STEP_RULE,
    TOT_THOUGHT_GUIDANCE,
    TOT_VALUE_FINAL_INSTRUCTION,
    TOT_VALUE_PARTIAL_INSTRUCTION,
    TOT_VALUE_PROMPT,
    TOT_VOTE_CANDIDATE,
    TOT_VOTE_PROMPT,
    TREE_OF_THOUGHTS_SYSTEM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.execution_utils import batched_run
from swarms.tools.base_tool import BaseTool
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import empty_usage
from swarms.utils.output_types import OutputType

SearchAlgorithm = Literal["bfs", "dfs"]
GenerationStrategy = Literal["propose", "sample"]
EvaluationStrategy = Literal["value", "vote"]

SchemaT = TypeVar("SchemaT", bound=BaseModel)
JobT = TypeVar("JobT")
ResultT = TypeVar("ResultT")


class Thought(BaseModel):
    """One candidate next step toward solving the task."""

    content: str = Field(
        ...,
        description=(
            "The next step, written so it can be read on its own. Build on "
            "the steps so far; do not repeat them."
        ),
    )
    is_final: bool = Field(
        False,
        description=(
            "True only when this step completes the task and states the "
            "full final answer."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data: Any) -> Any:
        """Accept a bare string, which some models send instead of an object."""
        if isinstance(data, str):
            return {"content": data}
        return data


class ProposeThoughts(BaseModel):
    """Submit candidate next steps that continue the partial solution."""

    thoughts: List[Thought] = Field(
        ...,
        description="Distinct candidate next steps, each a different way to continue.",
    )


class EvaluateThought(BaseModel):
    """Submit a rating of how promising a partial solution is."""

    # reasoning precedes score so the model critiques before it rates.
    reasoning: str = Field(
        ...,
        description=(
            "A short critique: is the latest step correct, and is this path "
            "on track to a correct, complete answer?"
        ),
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "0 = wrong or a dead end, 0.5 = plausible but uncertain, "
            "1 = certainly correct and on track. For a final answer, rate "
            "whether the answer is correct and complete."
        ),
    )


class VoteForThought(BaseModel):
    """Submit a vote for the most promising candidate."""

    reasoning: str = Field(
        ...,
        description="Compare the candidates and justify the choice.",
    )
    best_candidate: int = Field(
        ...,
        ge=1,
        description="The number of the most promising candidate, as listed.",
    )


class FinalAnswer(BaseModel):
    """Submit the final answer to the task."""

    answer: str = Field(
        ...,
        description=(
            "The complete final answer, written for whoever asked the task. "
            "It must stand on its own without referring to the steps."
        ),
    )


def _inline_refs(schema: Any, definitions: Dict[str, Any]) -> Any:
    """Replace local ``$ref`` pointers with their definitions and drop ``$defs``.

    Some providers ignore ``$ref`` in tool schemas; the model then invents its
    own field names for nested objects and every call fails validation.

    Args:
        schema: A JSON schema, or any value nested inside one.
        definitions: The schema's ``$defs``, keyed by definition name.

    Returns:
        The schema with every local reference expanded in place.
    """
    if isinstance(schema, list):
        return [_inline_refs(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _inline_refs(
            definitions[ref.rsplit("/", 1)[-1]], definitions
        )
    return {
        key: _inline_refs(value, definitions)
        for key, value in schema.items()
        if key != "$defs"
    }


def _tool_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Build the OpenAI function schema for ``model`` with no ``$ref`` left in it.

    Args:
        model: The Pydantic model describing the function's arguments.

    Returns:
        Dict[str, Any]: A tool definition named after the model class.
    """
    schema = BaseTool().base_model_to_dict(model)
    parameters = schema["function"]["parameters"]
    return {
        **schema,
        "function": {
            **schema["function"],
            "parameters": _inline_refs(
                parameters, parameters.get("$defs", {})
            ),
        },
    }


_TOOL_SCHEMAS: Dict[Type[BaseModel], Dict[str, Any]] = {
    model: _tool_schema(model)
    for model in (
        ProposeThoughts,
        EvaluateThought,
        VoteForThought,
        FinalAnswer,
    )
}


def _format_steps(steps: Sequence[str]) -> str:
    """Number reasoning steps one per line, or say there are none yet."""
    if not steps:
        return TOT_NO_STEPS
    return "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, 1)
    )


def _normalize(text: str) -> str:
    """Case- and whitespace-insensitive key used to drop duplicate thoughts."""
    return " ".join(text.lower().split())


def _decode(text: str) -> Any:
    """Decode model text as JSON, a Python literal, or the first JSON object in prose.

    Args:
        text: Raw text from the model or an agent's rendered output.

    Returns:
        The decoded value, or ``None`` when nothing decodes.
    """
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (
            ValueError,
            SyntaxError,
            TypeError,
            MemoryError,
            RecursionError,
        ):
            continue
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            pass
    return None


def _tool_arguments(
    output: Any, tool_name: str
) -> Optional[Dict[str, Any]]:
    """Find the arguments of a ``tool_name`` call in an agent's output.

    Args:
        output: What ``Agent.run`` returned: a list of tool-call dicts, a
            single tool-call dict, a string holding either, or the arguments
            as plain JSON when the model answered in text instead of calling.
        tool_name: The function whose arguments to return.

    Returns:
        The call's arguments, or ``None`` when no such call is present.
    """
    if isinstance(output, BaseModel):
        output = output.model_dump()
    if isinstance(output, str):
        output = _decode(output)
    for call in output if isinstance(output, list) else [output]:
        if isinstance(call, BaseModel):
            call = call.model_dump()
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if function is None:
            return call
        if not isinstance(function, dict) or function.get(
            "name"
        ) not in (None, tool_name):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = _decode(arguments)
        if isinstance(arguments, dict):
            return arguments
    return None


def parse_tool_call(
    output: Any, schema: Type[SchemaT]
) -> Optional[SchemaT]:
    """Validate the function call for ``schema`` found in an agent's output.

    The tool name is the schema's class name, as produced by
    ``BaseTool.base_model_to_dict``.

    Args:
        output: What ``Agent.run`` returned for a call made with the schema's
            tool attached.
        schema: The Pydantic model describing the function's arguments.

    Returns:
        The validated arguments, or ``None`` when the output holds no call
        or the call does not match the schema.
    """
    arguments = _tool_arguments(output, schema.__name__)
    if arguments is None:
        return None
    try:
        return schema.model_validate(arguments)
    except ValidationError:
        return None


# Identity equality: field-by-field comparison would recurse through every subtree.
@dataclass(eq=False)
class ThoughtNode:
    """One node of the search tree: a reasoning step and its evaluation.

    Attributes:
        content: The reasoning step. Empty for the root.
        depth: Number of steps from the root; the root is 0.
        parent: The node this step continues from; ``None`` for the root.
        is_final: Whether this step completes the task. Steps at
            ``max_depth`` are always final.
        score: Evaluation in ``[0, 1]``, or ``None`` until evaluated.
        evaluation: The evaluator's critique, or the vote tally.
        pruned: Whether the score fell below ``value_threshold``.
        children: Candidate next steps generated from this node.
    """

    content: str = ""
    depth: int = 0
    parent: Optional["ThoughtNode"] = field(default=None, repr=False)
    is_final: bool = False
    score: Optional[float] = None
    evaluation: Optional[str] = None
    pruned: bool = False
    children: List["ThoughtNode"] = field(
        default_factory=list, repr=False
    )

    @property
    def steps(self) -> List[str]:
        """The reasoning steps on the path from the root to this node."""
        steps = []
        node = self
        while node.parent is not None:
            steps.append(node.content)
            node = node.parent
        return steps[::-1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this node and its subtree, without parent links.

        Returns:
            Dict[str, Any]: JSON-serializable node data with nested children.
        """
        return {
            "content": self.content,
            "depth": self.depth,
            "is_final": self.is_final,
            "score": self.score,
            "evaluation": self.evaluation,
            "pruned": self.pruned,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class TreeOfThoughtsResult:
    """The outcome of one tree search.

    Attributes:
        task: The task that was searched.
        answer: The final answer written from the best path.
        solved: Whether the search reached a final step that cleared
            ``value_threshold``. When ``False``, ``answer`` was completed
            from the most promising partial path.
        best_node: The node the answer was written from, or ``None`` if no
            candidate was ever evaluated.
        root: The root of the search tree, for inspection.
        nodes_expanded: How many nodes had candidates generated from them.
        llm_calls: How many model calls the search made, including the final
            answer.
        usage: Provider token usage summed over this search's model calls.
    """

    task: str
    answer: str
    solved: bool
    best_node: Optional[ThoughtNode]
    root: ThoughtNode
    nodes_expanded: int
    llm_calls: int
    usage: Dict[str, int]

    @property
    def steps(self) -> List[str]:
        """The reasoning steps on the best path."""
        return self.best_node.steps if self.best_node else []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result, including the whole tree.

        Returns:
            Dict[str, Any]: JSON-serializable result data.
        """
        return {
            "task": self.task,
            "answer": self.answer,
            "solved": self.solved,
            "steps": self.steps,
            "nodes_expanded": self.nodes_expanded,
            "llm_calls": self.llm_calls,
            "usage": dict(self.usage),
            "tree": self.root.to_dict(),
        }


@dataclass
class _SearchState:
    """Per-search bookkeeping, kept off the agent so searches can run concurrently."""

    task: str
    root: ThoughtNode = field(default_factory=ThoughtNode)
    nodes_expanded: int = 0
    llm_calls: int = 0
    usage: Dict[str, int] = field(default_factory=empty_usage)
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def record(self, agent: Agent) -> None:
        """Count one model call and fold the agent's token usage in.

        Args:
            agent: The single-use agent that made the call.
        """
        with self.lock:
            self.llm_calls += 1
            for key, value in agent.usage.items():
                if key in self.usage:
                    self.usage[key] += value


class TreeOfThoughts:
    """Solve a task by searching a tree of reasoning steps.

    Each model call is a function call made by a fresh, stateless
    ``Agent`` that carries only that function's schema, so calls are
    independent of one another and those at the same level of the tree run
    concurrently. Output that holds no valid call is logged and skipped.

    Scores are in ``[0, 1]``. In ``"value"`` mode a score is the evaluator's
    rating, averaged over ``n_evaluate_samples`` calls. In ``"vote"`` mode it
    is a candidate's votes divided by the leading candidate's votes, so the
    leader always scores 1. Vote mode ranks candidates against each other,
    so it cannot reject a lone candidate, and with one vote per level only
    the winner survives; raise ``n_evaluate_samples`` to keep more branches.

    Candidates scoring below ``value_threshold`` are pruned: never expanded
    and never accepted as the answer. A final candidate that survives is a
    solution. BFS stops once its best solution scores at least as high as
    every open partial path. DFS visits children best-first and returns the
    first solution it reaches. If no solution is found, the answer is
    completed from the highest-scoring partial path.

    Reference:
        Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y.,
        & Narasimhan, K. (2023). Tree of Thoughts: Deliberate Problem
        Solving with Large Language Models. https://arxiv.org/abs/2305.10601

    Example:
        >>> agent = TreeOfThoughts(
        ...     model_name="gpt-5.4",
        ...     search_algorithm="dfs",
        ...     thought_description="One arithmetic operation on two of the remaining numbers.",
        ...     evaluation_criteria="Can the remaining numbers still reach 24?",
        ... )
        >>> print(agent.run("Use 4, 9, 10 and 13 with + - * / to make 24."))
        >>> print(agent.last_result.steps)
    """

    def __init__(
        self,
        name: str = "Tree-of-Thoughts-Agent",
        description: str = "Solves tasks by searching a tree of reasoning steps.",
        model_name: str = "gpt-5.4",
        system_prompt: str = TREE_OF_THOUGHTS_SYSTEM_PROMPT,
        search_algorithm: SearchAlgorithm = "bfs",
        generation_strategy: GenerationStrategy = "propose",
        evaluation_strategy: EvaluationStrategy = "value",
        num_thoughts: int = 3,
        breadth: int = 2,
        max_depth: int = 3,
        n_evaluate_samples: int = 1,
        value_threshold: float = 0.5,
        max_expansions: Optional[int] = None,
        thought_description: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        temperature: Optional[float] = None,
        max_workers: int = 8,
        output_type: OutputType = "final",
        verbose: bool = False,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the search.

        Args:
            name: Name used in logs, in the conversation, and as the
                prefix of every internal agent's name.
            description: Short description of the agent, for orchestrators.
            model_name: Any LiteLLM model string. The model must support
                function calling.
            system_prompt: System prompt for every model call. Replace it to
                give the search a domain persona; the step, evaluation and
                answer instructions are sent separately and stay in force.
            search_algorithm: ``"bfs"`` keeps the best ``breadth`` nodes per
                level; ``"dfs"`` follows the best child first and backtracks
                when a branch is pruned or exhausted.
            generation_strategy: ``"propose"`` asks for ``num_thoughts``
                distinct steps in one call, which suits constrained tasks.
                ``"sample"`` makes ``num_thoughts`` independent calls for one
                step each, which suits open-ended tasks.
            evaluation_strategy: ``"value"`` rates each candidate on its own;
                ``"vote"`` compares candidates and scores them by votes.
            num_thoughts: Candidate steps generated per expanded node.
            breadth: BFS beam width: nodes kept per level. Ignored by DFS.
            max_depth: Maximum number of steps on a path. Steps at this depth
                must complete the task.
            n_evaluate_samples: Evaluator calls per candidate (value) or per
                comparison (vote). More samples give steadier scores.
            value_threshold: Candidates scoring below this are pruned.
                Between 0 and 1.
            max_expansions: Cap on how many nodes may be expanded in one
                search, bounding cost. ``None`` means no cap; DFS can then
                expand up to ``num_thoughts ** (max_depth - 1)`` nodes and
                more.
            thought_description: What one step looks like for your task,
                e.g. "One SQL clause" or "A one-paragraph plan for the next
                section". Shown to the generator.
            evaluation_criteria: How to judge progress for your task, e.g.
                "Does the query still return the requested columns?". Shown
                to the evaluator.
            temperature: Sampling temperature for every call. ``None`` sends
                no temperature, so the provider's default applies; some
                models, such as Claude Sonnet 5, reject the parameter. If you
                set it, keep it above 0 when sampling or averaging several
                evaluations.
            max_workers: Maximum concurrent model calls. Set to 1 to run
                calls one at a time.
            output_type: How ``run`` formats its conversation. ``"final"``
                returns the answer string.
            verbose: Log every evaluated candidate.
            agent_kwargs: Extra ``Agent`` arguments applied to every internal
                call, such as ``max_tokens``, ``llm_api_key`` or
                ``llm_base_url``. Settings this class depends on
                (``tools_list_dictionary``, ``max_loops``, ``output_type``,
                ``system_prompt``) cannot be overridden.

        Raises:
            ValueError: If a strategy name is unknown or a numeric setting is
                out of range.
        """
        for setting, value, options in (
            ("search_algorithm", search_algorithm, SearchAlgorithm),
            (
                "generation_strategy",
                generation_strategy,
                GenerationStrategy,
            ),
            (
                "evaluation_strategy",
                evaluation_strategy,
                EvaluationStrategy,
            ),
        ):
            if value not in get_args(options):
                raise ValueError(
                    f"{setting} must be one of {get_args(options)}, got {value!r}"
                )
        for setting, value in (
            ("num_thoughts", num_thoughts),
            ("breadth", breadth),
            ("max_depth", max_depth),
            ("n_evaluate_samples", n_evaluate_samples),
            ("max_workers", max_workers),
        ):
            if value < 1:
                raise ValueError(
                    f"{setting} must be at least 1, got {value}"
                )
        if max_expansions is not None and max_expansions < 1:
            raise ValueError(
                f"max_expansions must be at least 1 or None, got {max_expansions}"
            )
        if not 0.0 <= value_threshold <= 1.0:
            raise ValueError(
                f"value_threshold must be between 0 and 1, got {value_threshold}"
            )

        self.name = name
        self.description = description
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.search_algorithm = search_algorithm
        self.generation_strategy = generation_strategy
        self.evaluation_strategy = evaluation_strategy
        self.num_thoughts = num_thoughts
        self.breadth = breadth
        self.max_depth = max_depth
        self.n_evaluate_samples = n_evaluate_samples
        self.value_threshold = value_threshold
        self.max_expansions = max_expansions
        self.thought_description = thought_description
        self.evaluation_criteria = evaluation_criteria
        self.temperature = temperature
        self.max_workers = max_workers
        self.output_type = output_type
        self.verbose = verbose
        self.agent_kwargs = dict(agent_kwargs or {})

        self.conversation = Conversation()
        self.last_result: Optional[TreeOfThoughtsResult] = None
        self._usage = empty_usage()
        self._usage_lock = threading.Lock()

    @property
    def usage(self) -> Dict[str, int]:
        """Provider token usage summed over every search this agent has run.

        Keys match :attr:`Agent.usage`: ``input_tokens``, ``output_tokens``,
        ``cached_tokens``, ``reasoning_tokens``, ``total_tokens``.

        Returns:
            Dict[str, int]: A new dict; mutating it does not affect the agent.
        """
        with self._usage_lock:
            return dict(self._usage)

    def run(self, task: str) -> Any:
        """Solve ``task`` with a tree search and return the answer.

        This is the agent's entry point. The conversation it formats holds
        the task, the best reasoning path, and the answer. The full search,
        with the tree, the best path and cost statistics, is kept on
        :attr:`last_result`.

        Args:
            task: The problem to solve.

        Returns:
            Any: The conversation formatted by ``output_type``; the answer
            string for the default ``"final"``.

        Raises:
            ValueError: If ``task`` is empty.
            RuntimeError: If the search found no solution and the final
                answer call also failed, so there is no answer to return.
        """
        result = self._search(task)
        self.last_result = result

        conversation = Conversation()
        conversation.add(role="User", content=task)
        conversation.add(
            role=self.name,
            content=f"Reasoning path:\n{_format_steps(result.steps)}",
        )
        conversation.add(role=self.name, content=result.answer)
        self.conversation = conversation

        return history_output_formatter(
            conversation, type=self.output_type
        )

    def _search(self, task: str) -> TreeOfThoughtsResult:
        """Run the tree search and write the final answer.

        Args:
            task: The problem to solve.

        Returns:
            TreeOfThoughtsResult: The answer, the best path, the whole tree,
            and cost statistics.

        Raises:
            ValueError: If ``task`` is empty.
            RuntimeError: If the search found no solution and the final
                answer call also failed.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        state = _SearchState(task=task)
        if self.search_algorithm == "bfs":
            solution = self._bfs(state)
        else:
            solution = self._dfs(state, state.root)

        best = solution or self._best_partial(state.root)
        answer = self._final_answer(
            state, best, solved=bool(solution)
        )

        with self._usage_lock:
            for key, value in state.usage.items():
                self._usage[key] += value

        return TreeOfThoughtsResult(
            task=task,
            answer=answer,
            solved=solution is not None,
            best_node=best,
            root=state.root,
            nodes_expanded=state.nodes_expanded,
            llm_calls=state.llm_calls,
            usage=dict(state.usage),
        )

    def _bfs(self, state: _SearchState) -> Optional[ThoughtNode]:
        """Beam search: expand the best ``breadth`` open nodes, level by level.

        Args:
            state: The current search.

        Returns:
            The best solution found, or ``None``.
        """
        frontier = [state.root]
        best_solution: Optional[ThoughtNode] = None
        for _ in range(self.max_depth):
            frontier = self._within_budget(state, frontier)
            if not frontier:
                break
            children = self._expand(state, frontier)
            self._evaluate(state, children)
            ranked = self._rank(children)
            for child in ranked:
                if child.is_final and (
                    best_solution is None
                    or child.score > best_solution.score
                ):
                    best_solution = child
            frontier = [c for c in ranked if not c.is_final][
                : self.breadth
            ]
            if best_solution is not None and (
                not frontier
                or best_solution.score >= frontier[0].score
            ):
                break
        return best_solution

    def _dfs(
        self, state: _SearchState, node: ThoughtNode
    ) -> Optional[ThoughtNode]:
        """Depth-first search that tries the best child first and backtracks.

        Args:
            state: The current search.
            node: The node to expand.

        Returns:
            The first solution reached below ``node``, or ``None``.
        """
        if node.depth >= self.max_depth or not self._within_budget(
            state, [node]
        ):
            return None
        children = self._expand(state, [node])
        self._evaluate(state, children)
        for child in self._rank(children):
            if child.is_final:
                return child
            solution = self._dfs(state, child)
            if solution is not None:
                return solution
        return None

    def _within_budget(
        self, state: _SearchState, nodes: List[ThoughtNode]
    ) -> List[ThoughtNode]:
        """Trim ``nodes`` to what ``max_expansions`` still allows.

        Args:
            state: The current search.
            nodes: Nodes about to be expanded, best first.

        Returns:
            The leading nodes that fit in the remaining budget.
        """
        if self.max_expansions is None:
            return nodes
        remaining = self.max_expansions - state.nodes_expanded
        return nodes[: max(0, remaining)]

    def _rank(self, nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        """Return the unpruned nodes, highest score first.

        Args:
            nodes: Evaluated nodes.

        Returns:
            Surviving nodes in descending score order; ties keep their order.
        """
        return sorted(
            (node for node in nodes if not node.pruned),
            key=lambda node: node.score,
            reverse=True,
        )

    def _expand(
        self, state: _SearchState, nodes: List[ThoughtNode]
    ) -> List[ThoughtNode]:
        """Generate and attach candidate children for every node, concurrently.

        Args:
            state: The current search.
            nodes: Nodes to expand.

        Returns:
            All new children, grouped by parent in the order of ``nodes``.
        """
        state.nodes_expanded += len(nodes)
        if self.generation_strategy == "propose":
            calls_per_node, thoughts_per_call = 1, self.num_thoughts
        else:
            calls_per_node, thoughts_per_call = self.num_thoughts, 1

        jobs = [node for node in nodes for _ in range(calls_per_node)]
        proposals = self._run_concurrently(
            lambda node: self._call(
                state,
                "generator",
                ProposeThoughts,
                self._generation_prompt(
                    state, node, thoughts_per_call
                ),
            ),
            jobs,
        )

        children: List[ThoughtNode] = []
        for node in nodes:
            seen = set()
            for job, proposal in zip(jobs, proposals):
                if job is not node or proposal is None:
                    continue
                for thought in proposal.thoughts[:thoughts_per_call]:
                    key = _normalize(thought.content)
                    if not key or key in seen:
                        continue
                    if len(node.children) == self.num_thoughts:
                        break
                    seen.add(key)
                    depth = node.depth + 1
                    node.children.append(
                        ThoughtNode(
                            content=thought.content.strip(),
                            depth=depth,
                            parent=node,
                            # The last allowed step is terminal whether or not the model flagged it.
                            is_final=thought.is_final
                            or depth == self.max_depth,
                        )
                    )
            children.extend(node.children)
        return children

    def _evaluate(
        self, state: _SearchState, nodes: List[ThoughtNode]
    ) -> None:
        """Score ``nodes`` in place and mark those below the threshold as pruned.

        Args:
            state: The current search.
            nodes: Freshly generated candidates.
        """
        if not nodes:
            return
        if self.evaluation_strategy == "value":
            self._evaluate_by_value(state, nodes)
        else:
            self._evaluate_by_vote(state, nodes)

        for node in nodes:
            node.pruned = node.score < self.value_threshold
            if self.verbose:
                flags = ("final " if node.is_final else "") + (
                    "pruned " if node.pruned else ""
                )
                logger.info(
                    f"[{self.name}] depth {node.depth} "
                    f"score {node.score:.2f} {flags}| {node.content[:120]}"
                )

    def _evaluate_by_value(
        self, state: _SearchState, nodes: List[ThoughtNode]
    ) -> None:
        """Rate each node on its own, averaging ``n_evaluate_samples`` ratings.

        A node whose every rating failed scores 0 and is pruned.

        Args:
            state: The current search.
            nodes: Candidates to score.
        """
        jobs = [
            node
            for node in nodes
            for _ in range(self.n_evaluate_samples)
        ]
        ratings = self._run_concurrently(
            lambda node: self._call(
                state,
                "evaluator",
                EvaluateThought,
                self._value_prompt(state, node),
            ),
            jobs,
        )
        for node in nodes:
            valid = [
                rating
                for job, rating in zip(jobs, ratings)
                if job is node and rating is not None
            ]
            if valid:
                node.score = sum(r.score for r in valid) / len(valid)
                node.evaluation = valid[0].reasoning
            else:
                node.score = 0.0
                node.evaluation = "Evaluation failed."

    def _evaluate_by_vote(
        self, state: _SearchState, nodes: List[ThoughtNode]
    ) -> None:
        """Score nodes by votes relative to the leader, so the leader scores 1.

        Args:
            state: The current search.
            nodes: Candidates to compare with one another.
        """
        if len(nodes) == 1:
            nodes[0].score = 1.0
            nodes[0].evaluation = "Only candidate."
            return

        prompt = self._vote_prompt(state, nodes)
        ballots = self._run_concurrently(
            lambda _: self._call(
                state, "evaluator", VoteForThought, prompt
            ),
            range(self.n_evaluate_samples),
        )
        votes = [0] * len(nodes)
        for ballot in ballots:
            if ballot is not None and ballot.best_candidate <= len(
                nodes
            ):
                votes[ballot.best_candidate - 1] += 1

        leader = max(votes)
        for node, count in zip(nodes, votes):
            node.score = count / leader if leader else 0.0
            node.evaluation = (
                f"{count} of {self.n_evaluate_samples} votes"
            )

    def _best_partial(
        self, root: ThoughtNode
    ) -> Optional[ThoughtNode]:
        """Find the highest-scoring evaluated node, preferring deeper ones on ties.

        Args:
            root: The root of the search tree.

        Returns:
            The best node, or ``None`` if nothing was evaluated.
        """
        best: Optional[ThoughtNode] = None
        stack = list(root.children)
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if node.score is None:
                continue
            if best is None or (node.score, node.depth) > (
                best.score,
                best.depth,
            ):
                best = node
        return best

    def _final_answer(
        self,
        state: _SearchState,
        best: Optional[ThoughtNode],
        solved: bool,
    ) -> str:
        """Write the final answer from the best path.

        Args:
            state: The current search.
            best: The node to answer from, or ``None``.
            solved: Whether ``best`` is an accepted solution.

        Returns:
            The final answer. If the answer call fails on a solved path, the
            solution step itself is returned.

        Raises:
            RuntimeError: If the answer call fails and the path is unsolved.
        """
        steps = best.steps if best else []
        if solved:
            status = TOT_FINAL_STATUS_SOLVED
        elif steps:
            status = TOT_FINAL_STATUS_PARTIAL
        else:
            status = TOT_FINAL_STATUS_EMPTY

        final = self._call(
            state,
            "answerer",
            FinalAnswer,
            TOT_FINAL_PROMPT.format(
                task=state.task,
                steps=_format_steps(steps),
                status=status,
                tool=FinalAnswer.__name__,
            ),
        )
        if final is not None and final.answer.strip():
            return final.answer.strip()
        if solved:
            return steps[-1]
        raise RuntimeError(
            f"[{self.name}] Tree of Thoughts produced no answer: the "
            "search found no solution and the final answer call failed. "
            "See the warnings above for the failing calls."
        )

    def _generation_prompt(
        self,
        state: _SearchState,
        node: ThoughtNode,
        count: int,
    ) -> str:
        """Build the prompt asking for ``count`` next steps after ``node``.

        Args:
            state: The current search.
            node: The node being expanded.
            count: How many candidates to ask for.

        Returns:
            The generator prompt.
        """
        if count == 1:
            candidates = TOT_ONE_CANDIDATE
        else:
            candidates = TOT_MANY_CANDIDATES.format(count=count)
        if node.depth + 1 >= self.max_depth:
            rule = TOT_LAST_STEP_RULE
        else:
            rule = TOT_STEP_RULE
        guidance = (
            TOT_THOUGHT_GUIDANCE.format(
                thought_description=self.thought_description
            )
            if self.thought_description
            else ""
        )
        return TOT_GENERATE_PROMPT.format(
            task=state.task,
            steps=_format_steps(node.steps),
            guidance=guidance,
            depth=node.depth + 1,
            max_depth=self.max_depth,
            candidates=candidates,
            rule=rule,
            tool=ProposeThoughts.__name__,
        )

    def _criteria(self) -> str:
        """Format ``evaluation_criteria`` as a prompt section, or nothing."""
        if not self.evaluation_criteria:
            return ""
        return TOT_EVALUATION_CRITERIA.format(
            evaluation_criteria=self.evaluation_criteria
        )

    def _value_prompt(
        self, state: _SearchState, node: ThoughtNode
    ) -> str:
        """Build the prompt asking the evaluator to rate ``node``'s path.

        Args:
            state: The current search.
            node: The candidate to rate.

        Returns:
            The value-evaluator prompt.
        """
        if node.is_final:
            instruction = TOT_VALUE_FINAL_INSTRUCTION
        else:
            instruction = TOT_VALUE_PARTIAL_INSTRUCTION
        return TOT_VALUE_PROMPT.format(
            task=state.task,
            steps=_format_steps(node.steps),
            criteria=self._criteria(),
            instruction=instruction,
            tool=EvaluateThought.__name__,
        )

    def _vote_prompt(
        self, state: _SearchState, nodes: List[ThoughtNode]
    ) -> str:
        """Build the prompt asking the evaluator to pick the best of ``nodes``.

        Args:
            state: The current search.
            nodes: The candidates, numbered from 1 in the prompt.

        Returns:
            The vote-evaluator prompt.
        """
        candidates = "\n\n".join(
            TOT_VOTE_CANDIDATE.format(
                index=index,
                marker=TOT_FINAL_MARKER if node.is_final else "",
                steps=_format_steps(node.steps),
            )
            for index, node in enumerate(nodes, 1)
        )
        return TOT_VOTE_PROMPT.format(
            task=state.task,
            candidates=candidates,
            criteria=self._criteria(),
            tool=VoteForThought.__name__,
        )

    def _call(
        self,
        state: _SearchState,
        role: str,
        schema: Type[SchemaT],
        prompt: str,
    ) -> Optional[SchemaT]:
        """Make one function call through a fresh, stateless agent.

        Args:
            state: The current search, which records the call and its usage.
            role: Suffix for the agent's name, e.g. ``"generator"``.
            schema: The function the model must call.
            prompt: The task sent to the agent.

        Returns:
            The validated call arguments, or ``None`` if the call failed or
            returned no valid call. Failures are logged as warnings.
        """
        agent = Agent(
            **{
                **self.agent_kwargs,
                "agent_name": f"{self.name}-{role}",
                "system_prompt": self.system_prompt,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_loops": 1,
                "tools_list_dictionary": [_TOOL_SCHEMAS[schema]],
                "output_type": "final",
                "print_on": False,
                "persistent_memory": False,
            }
        )
        # One failed call costs one candidate, not the whole search; the warning keeps it visible.
        try:
            output = agent.run(task=prompt)
        except Exception as error:
            logger.warning(
                f"[{self.name}] {role} call failed: "
                f"{type(error).__name__}: {error}"
            )
            return None
        finally:
            state.record(agent)

        parsed = parse_tool_call(output, schema)
        if parsed is None:
            logger.warning(
                f"[{self.name}] {role} did not return a valid "
                f"{schema.__name__} call: {str(output)[:200]}"
            )
        return parsed

    def _run_concurrently(
        self,
        func: Callable[[JobT], ResultT],
        jobs: Sequence[JobT],
    ) -> List[ResultT]:
        """Run ``func`` over ``jobs`` on up to ``max_workers`` threads.

        Args:
            func: Called once per job.
            jobs: The inputs.

        Returns:
            Results in the order of ``jobs``.
        """
        jobs = list(jobs)
        if not jobs:
            return []
        return batched_run(
            func, jobs, max_workers=min(self.max_workers, len(jobs))
        )
