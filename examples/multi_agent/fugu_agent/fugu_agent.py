"""
FuguAgent: Multi-agent orchestration system that behaves like a single model API.

The FuguAgent coordinates a pool of worker agents through a dedicated coordinator
model using tool-calling (not text parsing). At each step the coordinator calls
the ``decide_next_step`` tool, committing to a structured AgentTask containing
role, worker, instruction, and visibility. Workers are ranked by capability tier
to ensure the most powerful models handle the hardest tasks.

Example:
    >>> from examples.multi_agent.fugu_agent import FuguAgent
    >>> agent = FuguAgent(max_turns=5, verbose=True)
    >>> result = agent.run("Write a short story about AI.")
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent


MODEL_TIER: dict[str, int] = {
{
  "gpt-5": 10,
  "gpt-5-thinking": 10,
  "o4": 10,
  "o3": 10,

  "claude-opus-4.1": 10,
  "claude-opus-4": 9,
  "claude-sonnet-4": 9,
  "claude-sonnet-3.7": 8,
  "claude-sonnet-3.5": 7,

  "gemini-2.5-pro": 9,
  "gemini-2.5-flash": 7,
  "gemini-2.0-flash": 5,

  "grok-4": 9,
  "grok-4-heavy": 10,

  "gpt-4o": 7,
  "gpt-4-turbo": 6,
  "gpt-4": 6,
  "gpt-4o-mini": 5,
  "gpt-3.5-turbo": 3,

  "deepseek-r1": 8,
  "deepseek-v3.1": 8,
  "deepseek-v3": 7,

  "qwen3-235b-a22b-thinking": 8,
  "qwen3-235b-a22b-instruct": 7,
  "qwen3-32b": 6,
  "qwen3-8b": 4,
  "qwen2.5-72b": 6,

  "kimi-k2": 8,

  "magistral-medium": 7,
  "magistral-small": 5,

  "llama-4-maverick": 8,
  "llama-4-scout": 6,

  "command-a": 7,
  "command-r7b": 5,

  "gemma-3-27b": 5,
  "gemma-3-12b": 4,
  "ministral-8b": 4
}


def _model_tier(name: str) -> int:
    """
    Get the capability tier score for a model based on its name.

    Args:
        name: The model name to look up.

    Returns:
        int: Tier score from 1-10, defaults to 3 if unknown.
    """
    n = name.lower()
    for k, v in MODEL_TIER.items():
        if k in n:
            return v
    return 3


def _detect_models() -> list[str]:
    """
    Auto-detect available models from environment API keys.

    Returns:
        list of detected model names based on available API keys.
    """
    models: list[str] = []
    if os.getenv("OPENAI_API_KEY"):
        models.extend(["gpt-4o", "gpt-4o-mini"])
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append("claude-sonnet-4-5")
    if os.getenv("GOOGLE_API_KEY"):
        models.append("gemini-2.5-pro")
    return models


class AgentTask(BaseModel):
    """
    Represents a single task assigned to a worker agent.

    Attributes:
        role: The role for this task (e.g., 'planner', 'coder', 'reviewer').
        worker: The name of the worker agent to execute this task.
        instruction: The instruction prompt for the worker.
        visibility: List of prior step indices whose outputs the worker can see.
    """

    role: str = Field(description="Role assigned to this task (e.g., planner, coder)")
    worker: str = Field(description="Name of the worker agent to execute this task")
    instruction: str = Field(description="Instruction prompt for the worker")
    visibility: list[int] = Field(
        default_factory=list,
        description="Indices of prior step outputs visible to this worker",
    )


class AgentTaskResult(BaseModel):
    """
    Result of a completed task execution.

    Attributes:
        task: The AgentTask that was executed.
        output: The string output from the worker.
        artifacts: Optional dict of additional artifacts produced.
    """

    task: AgentTask = Field(description="The task that was executed")
    output: str = Field(description="Output string from the worker")
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional artifacts produced during execution",
    )


class VerificationResult(BaseModel):
    """
    Result of verification step.

    Attributes:
        confidence: Confidence score between 0 and 1.
        issues: List of issues found during verification.
        accept: Whether the work was accepted.
        diagnosis: Human-readable diagnosis summary.
    """

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Issues found during verification",
    )
    accept: bool = Field(description="Whether the work was accepted")
    diagnosis: str = Field(description="Human-readable diagnosis summary")


class WorkflowState:
    """
    Mutable state container for tracking workflow execution across turns.

    Attributes:
        tasks: List of all AgentTask objects created during the workflow.
        results: List of all AgentTaskResult objects from executed tasks.
        turn: Current turn number (0-indexed).
    """

    def __init__(self) -> None:
        """Initialize an empty workflow state."""
        self.tasks: list[AgentTask] = []
        self.results: list[AgentTaskResult] = []
        self.turn: int = 0

    def __repr__(self) -> str:
        return (
            f"WorkflowState(turn={self.turn}, "
            f"tasks={len(self.tasks)}, results={len(self.results)})"
        )


class MemoryStore:
    """
    SQLite-backed persistent memory store for workflow sessions.

    Stores task artifacts and metadata across turns and sessions using an
    SQLite database for durability.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize the memory store.

        Args:
            db_path: Optional path to SQLite db. Defaults to /tmp/fugu_memory.db.
        """
        self.db_path: str = db_path or "/tmp/fugu_memory.db"
        self._init_db()

    def _init_db(self) -> None:
        """Create the memory table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    artifact TEXT,
                    timestamp REAL,
                    tags TEXT
                )
                """
            )

    def save(self, task_id: str, artifact: Any, tags: str = "") -> None:
        """
        Save an artifact to memory.

        Args:
            task_id: Identifier for the task session.
            artifact: The artifact to store (must be JSON-serializable).
            tags: Optional comma-separated tags for the artifact.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory (task_id, artifact, timestamp, tags) VALUES (?, ?, ?, ?)",
                (task_id, json.dumps(artifact), time.time(), tags),
            )

    def search(self, query: str = "", k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve the k most recent artifacts from memory.

        Args:
            query: Unused, kept for API compatibility.
            k: Maximum number of artifacts to return.

        Returns:
            List of dicts with 'artifact', 'timestamp', and 'tags' keys.
        """
        del query
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT artifact, timestamp, tags
                FROM memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (k,),
            ).fetchall()
        return [
            {"artifact": json.loads(r[0]), "timestamp": r[1], "tags": r[2]}
            for r in rows
        ]

    def clear(self) -> None:
        """Delete all artifacts from memory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory")

    def __repr__(self) -> str:
        return f"MemoryStore(db_path={self.db_path!r})"


def _rank_workers(
    workers: list[tuple[str, Agent]]
) -> list[tuple[str, Agent, int]]:
    """
    Rank workers by their model's capability tier.

    Args:
        workers: List of (name, agent) tuples to rank.

    Returns:
        List of (name, agent, tier) tuples sorted by tier descending.
    """
    return sorted(
        [(name, agent, _model_tier(agent.model_name)) for name, agent in workers],
        key=lambda x: x[2],
        reverse=True,
    )


def _make_decide_tool(
    worker_pool: dict[str, Agent],
    ranked_workers: list[tuple[str, Agent, int]],
    task_holder: list[AgentTask],
) -> Callable[[str, str, str, list[int]], str]:
    """
    Create the ``decide_next_step`` tool for the coordinator.

    This tool allows the coordinator to commit to a structured next action
    by storing the AgentTask in the task_holder via closure capture.

    Args:
        worker_pool: Dict mapping worker names to Agent instances.
        ranked_workers: Workers sorted by capability tier.
        task_holder: List used to capture the decided task.

    Returns:
        A callable that implements the decide_next_step tool.
    """

    def decide_next_step(
        role: str,
        worker: str,
        instruction: str,
        visibility: list[int],
    ) -> str:
        """
        Decide the next action in the multi-agent workflow.

        Args:
            role: Role for this task (e.g., 'planner', 'coder').
            worker: Name of the worker to assign.
            instruction: Instruction prompt for the worker.
            visibility: List of prior step indices visible to the worker.

        Returns:
            JSON string representation of the decided AgentTask.
        """
        if worker not in worker_pool:
            worker = ranked_workers[0][0]
        task = AgentTask(
            role=role,
            worker=worker,
            instruction=instruction,
            visibility=visibility,
        )
        task_holder.clear()
        task_holder.append(task)
        return task.model_dump_json()

    return decide_next_step


def _build_coordinator_system_prompt(
    ranked_workers: list[tuple[str, Agent, int]]
) -> str:
    """
    Build the system prompt for the coordinator agent.

    Args:
        ranked_workers: Workers sorted by capability tier.

    Returns:
        A formatted system prompt string for the coordinator.
    """
    lines = [
        "You are the Coordinator for a multi-agent system.",
        "",
        "AVAILABLE WORKERS (ranked by capability — higher tier = more powerful):",
    ]
    for name, agent, tier in ranked_workers:
        lines.append(f"  [{tier}] {name} ({agent.model_name})")

    lines.extend([
        "",
        "DYNAMIC ROLES: Assign whichever role best fits the needed function:",
        "  planner, coder, researcher, writer, reviewer, summarizer, executor, etc.",
        "",
        "Use the decide_next_step tool to commit to the next action.",
        "The tool returns the structured AgentTask for immediate execution.",
        "Match higher-tier workers to harder/more critical subtasks.",
    ])
    return "\n".join(lines)


class FuguAgent:
    """
    Multi-agent orchestration system that behaves like a single model API.

    The FuguAgent coordinates a pool of worker agents through a dedicated
    coordinator model using tool-calling. At each step the coordinator decides
    which worker to use, what role to assign, and what instruction to give.
    Workers are ranked by capability and the most powerful models are assigned
    to the hardest subtasks.

    Attributes:
        coordinator_model: Model name for the coordinator agent.
        max_turns: Maximum number of workflow turns before terminating.
        confidence_threshold: Minimum confidence to accept a verification.
        verbose: Whether to print verbose progress information.
        memory: MemoryStore instance for persistence across sessions.
        coordinator: The coordinator Agent instance.
        worker_pool: Dict mapping worker names to Agent instances.
    """

    def __init__(
        self,
        coordinator_model: str = "gpt-4o-mini",
        workers: list[Agent] | None = None,
        worker_models: list[str] | None = None,
        max_turns: int = 5,
        confidence_threshold: float = 0.85,
        verbose: bool = False,
        memory_db_path: str | None = None,
    ) -> None:
        """
        Initialize a FuguAgent.

        Args:
            coordinator_model: Model name for the coordinator (default: gpt-4o-mini).
            workers: Explicit list of Agent instances to use as workers.
            worker_models: List of model names to auto-create workers from.
            max_turns: Maximum workflow turns (default: 5).
            confidence_threshold: Min confidence for verification acceptance (default: 0.85).
            verbose: Enable verbose output (default: False).
            memory_db_path: Optional path for the SQLite memory database.

        Raises:
            ValueError: If no workers, worker_models, or API keys are available.
        """
        self.coordinator_model: str = coordinator_model
        self.max_turns: int = max_turns
        self.confidence_threshold: float = confidence_threshold
        self.verbose: bool = verbose
        self.memory: MemoryStore = MemoryStore(db_path=memory_db_path)
        self._decide_holder: list[AgentTask] = []
        self._last_decided_task: AgentTask | None = None

        if workers is not None and len(workers) > 0:
            self.worker_pool: dict[str, Agent] = {
                w.agent_name: w for w in workers
            }
        elif worker_models is not None and len(worker_models) > 0:
            self.worker_pool = self._init_workers(worker_models)
        else:
            detected = _detect_models()
            if not detected:
                raise ValueError(
                    "No workers provided and no API keys detected. "
                    "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY, "
                    "or pass workers / worker_models explicitly."
                )
            self.worker_pool = self._init_workers(detected)

        self._ranked_workers: list[tuple[str, Agent, int]] = _rank_workers(
            list(self.worker_pool.items())
        )

        decide_tool = _make_decide_tool(
            self.worker_pool, self._ranked_workers, self._decide_holder
        )

        self.coordinator: Agent = Agent(
            agent_name="fugu_coordinator",
            model_name=self.coordinator_model,
            system_prompt=_build_coordinator_system_prompt(self._ranked_workers),
            max_loops=1,
            tools=[decide_tool],
        )

        if self.verbose:
            for name, agent, tier in self._ranked_workers:
                print(f"[FuguAgent] [{tier}] {name} = {agent.model_name}")

    def _init_workers(self, models: list[str]) -> dict[str, Agent]:
        """
        Initialize worker agents from a list of model names.

        Args:
            models: List of model names to create workers from.

        Returns:
            Dict mapping worker names to Agent instances.
        """
        specialties = [
            "general",
            "coder",
            "researcher",
            "math",
            "writer",
            "reviewer",
        ]
        pool: dict[str, Agent] = {}
        seen: set[str] = set()

        for i, model in enumerate(models):
            spec = specialties[i] if i < len(specialties) else f"worker_{i}"
            name = spec
            c = 0
            while name in seen:
                name = f"{spec}_{c}"
                c += 1
            seen.add(name)
            pool[name] = Agent(
                agent_name=f"fugu_{name}",
                model_name=model,
                system_prompt=f"You are a {name} specialist.",
                max_loops=1,
            )
        return pool

    def _decide_and_store(self) -> AgentTask | None:
        """
        Retrieve the last decided task from the coordinator's tool call.

        Returns:
            The decided AgentTask or None if no task was decided.
        """
        if self._decide_holder:
            task = self._decide_holder[0]
            self._last_decided_task = task
            self._decide_holder.clear()
            return task
        return self._last_decided_task

    def _build_context(self, state: WorkflowState) -> str:
        """
        Build a summary string of completed steps for the coordinator.

        Args:
            state: Current workflow state.

        Returns:
            A formatted string summarizing prior results.
        """
        if not state.results:
            return "No prior steps yet."
        return "\n".join(
            f"Step {i} [{r.task.role}][{r.task.worker}]: "
            f"{r.output[:200].replace(chr(10), ' ')}"
            for i, r in enumerate(state.results)
        )

    def _execute(self, task: AgentTask, state: WorkflowState) -> AgentTaskResult:
        """
        Execute a task using the assigned worker.

        Args:
            task: The AgentTask to execute.
            state: Current workflow state for building visibility context.

        Returns:
            An AgentTaskResult containing the worker's output.
        """
        ctx_parts: list[str] = []
        for idx in task.visibility:
            if idx < len(state.results):
                ctx_parts.append(f"[Step {idx}]: {state.results[idx].output}")
        ctx = "\n\n".join(ctx_parts)

        prompt = task.instruction
        if ctx:
            prompt = f"Prior context:\n{ctx}\n\nYour task:\n{task.instruction}"

        worker = self.worker_pool.get(
            task.worker, list(self.worker_pool.values())[0]
        )
        output = worker.run(prompt)

        if self.verbose:
            print(f"[{task.role}/{task.worker}] {output[:120]}")

        return AgentTaskResult(task=task, output=output)

    def _verify(
        self, state: WorkflowState, original: str
    ) -> VerificationResult:
        """
        Run verification on the accumulated work.

        Args:
            state: Current workflow state.
            original: The original user task.

        Returns:
            A VerificationResult with accept/revise decision.
        """
        accumulated = "\n".join(r.output for r in state.results)
        verifier = list(self.worker_pool.values())[-1]
        output = verifier.run(
            f"Task: {original}\n\nWork so far:\n{accumulated[:2000]}\n\n"
            "Evaluate. Output exactly one of:\n"
            "  ACCEPT — task is solved\n"
            "  REVISE: <one sentence describing what to fix>"
        )
        low = output.lower()
        if "accept" in low and "revise:" not in low.split("accept")[0][-10:]:
            return VerificationResult(
                confidence=0.95,
                issues=[],
                accept=True,
                diagnosis=output[:200],
            )
        diagnosis = (
            output[output.lower().find("revise:") + 7 :].strip()[:200]
            if "revise:" in low
            else output[:200]
        )
        return VerificationResult(
            confidence=0.3,
            issues=[diagnosis],
            accept=False,
            diagnosis=diagnosis,
        )

    def _aggregate(self, state: WorkflowState, original: str) -> str:
        """
        Synthesize all step outputs into a final answer.

        Args:
            state: Current workflow state.
            original: The original user task.

        Returns:
            A string containing the final synthesized answer.
        """
        if not state.results:
            return "No output generated."
        steps = "\n\n".join(
            f"--- Step {i} [{r.task.role}] ---\n{r.output}"
            for i, r in enumerate(state.results)
        )
        return self.coordinator.run(
            f"Task: {original}\n\n"
            f"Synthesize all step outputs into one final answer:\n\n{steps}"
        )

    def run(self, task: str) -> str:
        """
        Execute the workflow on a task string.

        Args:
            task: The user task to process.

        Returns:
            A string containing the final synthesized answer.
        """
        state = WorkflowState()
        session_id = f"session_{int(time.time())}"
        self.memory.save(session_id, {"task": task, "type": "input"})

        while state.turn < self.max_turns:
            self._decide_holder.clear()
            history_ctx = self._build_context(state)
            mem_ctx = json.dumps([r["artifact"] for r in self.memory.search(k=3)])

            self.coordinator.run(
                f"Original task: {task}\n\n"
                f"History:\n{history_ctx}\n\n"
                f"Memory: {mem_ctx}\n\n"
                "Use decide_next_step to decide the next action."
            )

            agent_task = self._decide_and_store()
            if agent_task is None:
                if self.verbose:
                    print("[Coordinator] Could not determine next step, ending.")
                break

            if agent_task.role in ("verifier", "reviewer"):
                ver = self._verify(state, task)
                state.results.append(
                    AgentTaskResult(
                        task=agent_task,
                        output=f"[VERIFIED] {ver.diagnosis}",
                    )
                )
                state.tasks.append(agent_task)
                if ver.accept:
                    break
            else:
                result = self._execute(agent_task, state)
                state.results.append(result)
                state.tasks.append(agent_task)

            state.turn += 1
            self.memory.save(
                session_id,
                {
                    "turn": state.turn,
                    "role": agent_task.role,
                    "worker": agent_task.worker,
                },
            )

        if self.verbose:
            print(f"[FuguAgent] Finished in {state.turn} turns")

        return self._aggregate(state, task)

    def __repr__(self) -> str:
        return (
            f"FuguAgent("
            f"coordinator_model={self.coordinator_model!r}, "
            f"max_turns={self.max_turns}, "
            f"workers={list(self.worker_pool.keys())}"
            f")"
        )
