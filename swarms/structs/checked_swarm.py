import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.context_utils import agent_answer
from swarms.structs.conversation import Conversation
from swarms.structs.execution_utils import batched_run
from swarms.structs.phased_fan_out_workflow import fresh_agent
from swarms.structs.usage_utils import total_usage
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)
from swarms.utils.generate_id import generate_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import OutputType

Check = Callable[[str], Union[bool, Tuple[bool, str]]]

STATUS_PROMPT = """Your answer is accepted only if an automatic check passes on it.
End your reply with exactly one status line and nothing after it:
STATUS: DONE when you believe your answer is complete and correct.
STATUS: STUCK: <what blocks you> when it is not.
An agent that says DONE and fails the check is replaced at once.
An agent that says STUCK gets the reason and another try."""

STATUS_LINE = re.compile(
    r"^[\s*_`>#]*STATUS:\s*(DONE|STUCK)\b", re.IGNORECASE
)

LESSONS = {
    "false_claim": 'An earlier agent on "{part}" claimed done but the check failed: {reason}',
    "stuck": 'An earlier agent on "{part}" could not pass the check: {reason}',
}


def _run_check(check: Check, answer: str) -> Tuple[bool, str]:
    """Return (passed, reason); a check that raises fails with its error."""
    try:
        result = check(answer)
        if not isinstance(result, (tuple, list)):
            result = (result, "")
        passed, reason = result
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    if passed:
        return True, ""
    return False, str(reason) or "The check failed."


def _split_status(reply: Any) -> Tuple[str, bool]:
    """Return the reply without its last status line, and whether it said DONE."""
    lines = str(reply or "").splitlines()
    said_done = False
    for index in reversed(range(len(lines))):
        match = STATUS_LINE.match(lines[index])
        if match:
            said_done = match.group(1).upper() == "DONE"
            del lines[index]
            break
    return "\n".join(lines).strip(), said_done


class CheckedSwarm:
    """One agent per part; an answer counts only if its check passes.

    The check decides whether an answer is accepted. The agent's status
    line decides what happens after a failed check: an agent that said
    DONE is retired at once, and an agent that did not gets the reason
    back and tries again, until max_attempts failures retire it as stuck.
    A retired agent is replaced by a fresh copy of its template that never
    sees the retired answer, only a one-line lesson that every part reads.

    Args:
        id: Unique identifier; generated when omitted.
        name: Name of the swarm.
        description: Description of the swarm.
        agents: One template agent per part, copied fresh for every use.
        checks: One check per part. A check takes the answer and returns
            a bool or a (passed, reason) tuple.
        parts: What each agent owns; defaults to "Part i of the task".
        max_attempts: Failed checks in a row, without a DONE claim,
            that retire an agent as stuck.
        max_generations: Agents a part may use before the part fails.
        output_type: Format of the value returned by run.
        verbose: Whether to log every event.

    Raises:
        ValueError: If the configuration is invalid.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "CheckedSwarm",
        description: str = "Accepts an agent's answer only when its check passes.",
        agents: Optional[List[Agent]] = None,
        checks: Optional[List[Check]] = None,
        parts: Optional[List[str]] = None,
        max_attempts: int = 3,
        max_generations: int = 2,
        output_type: OutputType = "dict-all-except-first",
        verbose: bool = False,
    ):
        self.id = id or generate_id("checked-swarm")
        self.name = name
        self.description = description
        self.agents = list(agents or [])
        self.checks = list(checks or [])
        self.parts = list(parts or []) or [
            f"Part {i} of the task"
            for i in range(1, len(self.agents) + 1)
        ]
        self.max_attempts = max_attempts
        self.max_generations = max_generations
        self.output_type = output_type
        self.verbose = verbose
        self.last_run: List[Dict[str, Any]] = []
        self.workers: List[Any] = []
        self._lock = threading.Lock()

        self.reliability_check()
        self.conversation = Conversation(name=self.name)
        capture_init(self)

    def reliability_check(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If there are no agents, the checks or parts do not
                match the agents one to one, or a limit is below 1.
        """
        if not self.agents:
            raise ValueError("CheckedSwarm needs at least one agent")
        if len(self.checks) != len(self.agents) or not all(
            callable(check) for check in self.checks
        ):
            raise ValueError("Give one callable check per agent")
        if len(self.parts) != len(self.agents):
            raise ValueError("Give one part per agent")
        if self.max_attempts < 1 or self.max_generations < 1:
            raise ValueError(
                "max_attempts and max_generations must be at least 1"
            )

    def _record(
        self, part: str, generation: int, event: str, reason: str
    ) -> None:
        """Add one event row to last_run."""
        row = {
            "part": part,
            "generation": generation,
            "event": event,
            "reason": reason,
        }
        with self._lock:
            self.last_run.append(row)
        if self.verbose:
            logger.info(f"{self.name}: {row}")

    def _prompt(self, task: str, part: str, feedback: str) -> str:
        """Build an attempt's prompt with the run's newest five lessons."""
        with self._lock:
            rows = [r for r in self.last_run if r["event"] in LESSONS]
        lessons = [
            " ".join(LESSONS[r["event"]].format(**r).split())
            for r in rows[-5:]
        ]
        sections = [
            f"Task:\n{task}",
            f"Your part, and the only part you answer:\n{part}",
        ]
        if lessons:
            sections.append(
                "Lessons from earlier agents in this run:\n- "
                + "\n- ".join(lessons)
            )
        if feedback:
            sections.append(f"Your last attempt failed: {feedback}")
        return "\n\n".join(sections + [STATUS_PROMPT])

    def _run_part(self, index: int, task: str) -> Tuple[str, str]:
        """Return (agent name, verified answer or "Unverified: <reason>")."""
        part, check = self.parts[index], self.checks[index]
        reason = ""
        for generation in range(1, self.max_generations + 1):
            agent = fresh_agent(self.agents[index])
            with self._lock:
                self.workers.append(agent)
            feedback = ""
            for attempt in range(1, self.max_attempts + 1):
                prompt = self._prompt(task, part, feedback)
                try:
                    reply = agent_answer(agent, agent.run(prompt))
                except Exception as error:
                    said_done, passed = False, False
                    reason = f"The agent raised {type(error).__name__}: {error}"
                else:
                    answer, said_done = _split_status(reply)
                    passed, reason = _run_check(check, answer)
                if passed:
                    self._record(part, generation, "verified", "")
                    return agent.agent_name, answer
                if said_done:
                    event = "false_claim"
                elif attempt < self.max_attempts:
                    event = "retry"
                else:
                    event = "stuck"
                self._record(part, generation, event, reason)
                if event != "retry":
                    break
                feedback = reason
        self._record(part, self.max_generations, "failed", reason)
        return self.agents[index].agent_name, f"Unverified: {reason}"

    @property
    def usage(self) -> dict:
        """Provider token usage of the last run, retired agents included.

        Returns:
            dict: Summed token counts in the shape of Agent usage.
        """
        return total_usage(self.workers)

    @trace_run(
        "CheckedSwarm.run",
        input_params=("task",),
        usage=lambda swarm: swarm.usage,
    )
    def run(self, task: str) -> Any:
        """Run every part at once and keep only answers that pass their check.

        Args:
            task: The task the parts belong to.

        Returns:
            Any: The conversation formatted by output_type. A failed part
            appears as "Unverified: <reason>", never as an answer.
        """
        self.last_run, self.workers = [], []
        self.conversation.clear()
        self.conversation.add("user", task)
        with ContextThreadPoolExecutor(len(self.agents)) as pool:
            futures = [
                pool.submit(self._run_part, index, task)
                for index in range(len(self.agents))
            ]
            for future in futures:
                self.conversation.add(*future.result())
        return history_output_formatter(
            self.conversation, type=self.output_type
        )

    def batched_run(self, tasks: List[str]) -> List[Any]:
        """Run several tasks one after another.

        Args:
            tasks: The tasks to run.

        Returns:
            List[Any]: One result per task.
        """
        return batched_run(self.run, tasks)

    def __call__(self, task: str) -> Any:
        """Make the swarm callable; same as run."""
        return self.run(task)
