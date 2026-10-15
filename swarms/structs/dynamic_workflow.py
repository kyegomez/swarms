import copy
import inspect
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from swarms.prompts.dynamic_workflow_prompts import (
    DYNAMIC_WORKFLOW_LEAD_PROMPT,
    DYNAMIC_WORKFLOW_SIZE_GUIDELINES as SIZE_GUIDELINES,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM, empty_usage
from swarms.utils.output_types import OutputType

PLACEHOLDER = re.compile(r"\{([A-Za-z_][\w\-]*)\}")


class DynamicWorkflowError(Exception):
    """Raised when a dynamic workflow cannot plan or finish its run."""


class InlineAgentSpec(BaseModel):
    name: str = Field(description="Unique agent name used by steps.")
    system_prompt: str = Field(
        description="System prompt for the agent."
    )


class Step(BaseModel):
    id: str = Field(description="Unique step id.")
    agent: str = Field(
        description="Name of a predefined or inline agent."
    )
    prompt: str = Field(
        description="Prompt; may use {step_id} for an earlier step's output and {item} in a fan-out."
    )
    for_each: Optional[str] = Field(
        default=None,
        description="'<step_id>.<field>': run one agent per item of that list.",
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema the agent's output must follow.",
    )


class Phase(BaseModel):
    title: str
    steps: List[Step] = Field(
        description="Steps in a phase run in parallel."
    )


class WorkflowPlan(BaseModel):
    name: str
    description: str
    inline_agents: List[InlineAgentSpec] = Field(default_factory=list)
    phases: List[Phase] = Field(description="Phases run in order.")
    return_steps: List[str] = Field(
        description="Step ids whose outputs the lead combines."
    )


def _parse_json(raw: Any) -> Any:
    """Parse JSON from an LLM reply, tolerating code fences and surrounding prose.

    Args:
        raw: The reply, a string or an already parsed value.

    Returns:
        Any: The parsed JSON value.
    """
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min(
            (i for i in (text.find("{"), text.find("[")) if i != -1),
            default=-1,
        )
        if start == -1:
            raise
        return json.loads(text[start:])


def fresh_agent(agent: Any) -> Any:
    """Build a new agent with the same configuration and no conversation state.

    Agents hold locks and cannot be deep-copied, so an Agent is rebuilt from
    its constructor arguments; the LLM client it built for itself is left out
    so the copy builds its own. Other agent-like objects are deep-copied.

    Args:
        agent: The agent to copy.

    Returns:
        Any: The new agent, with output_type set to final.
    """
    if isinstance(agent, Agent):
        params = inspect.signature(type(agent).__init__).parameters
        config = {
            name: getattr(agent, name)
            for name, param in params.items()
            if name != "self"
            and param.kind
            not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            and hasattr(agent, name)
        }
        if isinstance(config.get("llm"), LiteLLM):
            config.pop("llm")
        fresh = type(agent)(**config)
    else:
        fresh = copy.deepcopy(agent)
    fresh.output_type = "final"
    return fresh


def _as_text(value: Any) -> str:
    """Render a step output for insertion into a prompt.

    Args:
        value: The step output.

    Returns:
        str: The value itself for strings, JSON otherwise.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)


class DynamicWorkflow:
    """A lead agent plans a phased multi-agent run, the plan executes, and the lead combines the results.

    Args:
        name: Name of the workflow.
        description: Description of the workflow.
        lead_agent: Agent that plans and combines; built from model_name when omitted.
        model_name: Model for the default lead agent.
        agents: Predefined agents a plan may use, copied fresh for every use.
        allow_inline_agents: Whether the plan may define its own agents.
        worker_model_name: Model for inline agents; defaults to the lead's model.
        max_agents: Hard limit on worker agents started in one run.
        max_concurrent_agents: Agents that may run at the same time within a phase.
        size_guideline: small, medium, large or unrestricted; tells the lead how many agents to aim for.
        max_retries: Retries for a failed agent before its step fails.
        max_plan_attempts: Plans the lead may write before the run gives up.
        output_type: Format of the value returned by run.
    """

    def __init__(
        self,
        name: str = "DynamicWorkflow",
        description: str = "A lead agent plans a phased multi-agent run and combines the results.",
        lead_agent: Optional[Agent] = None,
        model_name: str = "gpt-5.4",
        agents: Optional[List[Agent]] = None,
        allow_inline_agents: bool = True,
        worker_model_name: Optional[str] = None,
        max_agents: int = 1000,
        max_concurrent_agents: int = 16,
        size_guideline: str = "medium",
        max_retries: int = 2,
        max_plan_attempts: int = 3,
        output_type: OutputType = "final",
    ):
        self.name = name
        self.description = description
        self.model_name = model_name
        self.lead_agent = lead_agent or Agent(
            agent_name=f"{name}-lead",
            agent_description="Plans the workflow and combines its results.",
            system_prompt=DYNAMIC_WORKFLOW_LEAD_PROMPT,
            model_name=model_name,
            max_loops=1,
            output_type="final",
        )
        self.agents = list(agents or [])
        self.allow_inline_agents = allow_inline_agents
        self.worker_model_name = worker_model_name or getattr(
            self.lead_agent, "model_name", model_name
        )
        self.max_agents = max_agents
        self.max_concurrent_agents = max_concurrent_agents
        self.size_guideline = size_guideline
        self.max_retries = max_retries
        self.max_plan_attempts = max_plan_attempts
        self.output_type = output_type
        self.last_plan: Optional[WorkflowPlan] = None
        self.last_run: Optional[Dict[str, Any]] = None

        self.reliability_check()
        capture_init(self)

    def reliability_check(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If a limit is out of range, the size guideline is unknown, or agent names repeat.
        """
        if self.max_agents < 1:
            raise ValueError("max_agents must be at least 1")
        if not 1 <= self.max_concurrent_agents <= 256:
            raise ValueError(
                "max_concurrent_agents must be between 1 and 256"
            )
        if self.size_guideline not in SIZE_GUIDELINES:
            raise ValueError(
                f"size_guideline must be one of {list(SIZE_GUIDELINES)}"
            )
        if self.max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        if self.max_plan_attempts < 1:
            raise ValueError("max_plan_attempts must be at least 1")
        names = [agent.agent_name for agent in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("Predefined agents need unique names")
        if not self.agents and not self.allow_inline_agents:
            raise ValueError(
                "Give predefined agents or allow inline agents"
            )

    def _fresh_lead(self) -> Agent:
        """Copy of the lead agent with no state from earlier runs.

        Returns:
            Agent: The copied lead.
        """
        return fresh_agent(self.lead_agent)

    def _build_agent(self, name: str, plan: WorkflowPlan) -> Agent:
        """Build a fresh agent for one use.

        Args:
            name: Name of a predefined or inline agent.
            plan: The plan that defines the inline agents.

        Returns:
            Agent: A new agent instance.
        """
        for agent in self.agents:
            if agent.agent_name == name:
                return fresh_agent(agent)
        spec = next(a for a in plan.inline_agents if a.name == name)
        return Agent(
            agent_name=spec.name,
            system_prompt=spec.system_prompt,
            model_name=self.worker_model_name,
            max_loops=1,
            output_type="final",
        )

    def _planning_prompt(self, task: str) -> str:
        """Prompt asking the lead for a plan.

        Args:
            task: The task to plan for.

        Returns:
            str: The planning prompt.
        """
        if self.agents:
            roster = "\n".join(
                f"- {a.agent_name}: {a.agent_description}"
                for a in self.agents
            )
        else:
            roster = "(none)"
        inline = (
            "You may define inline agents."
            if self.allow_inline_agents
            else "You may NOT define inline agents; use only the predefined agents."
        )
        return (
            f"Task:\n{task}\n\n"
            f"Predefined agents:\n{roster}\n\n"
            f"{inline}\n"
            f"Hard limit: {self.max_agents} worker agents per run. "
            f"{SIZE_GUIDELINES[self.size_guideline]}\n\n"
            "Return ONLY a JSON object matching this schema, with no other text:\n"
            f"{json.dumps(WorkflowPlan.model_json_schema())}"
        )

    def validate_plan(self, plan: WorkflowPlan) -> List[str]:
        """Check a plan before any agent runs.

        Args:
            plan: The plan to check.

        Returns:
            List[str]: The problems found; empty when the plan is valid.
        """
        errors = []
        predefined = {a.agent_name for a in self.agents}
        inline = [a.name for a in plan.inline_agents]
        if inline and not self.allow_inline_agents:
            errors.append(
                "Inline agents are not allowed in this run."
            )
        if len(inline) != len(set(inline)):
            errors.append("Inline agent names must be unique.")
        clashes = predefined.intersection(inline)
        if clashes:
            errors.append(
                f"Inline agents reuse predefined names: {sorted(clashes)}."
            )
        known_agents = predefined.union(inline)

        if not plan.phases:
            errors.append("The plan has no phases.")
        seen_ids = set()
        phase_of = {}
        schema_of = {}
        for index, phase in enumerate(plan.phases):
            if not phase.steps:
                errors.append(f"Phase '{phase.title}' has no steps.")
            for step in phase.steps:
                if step.id in seen_ids or step.id in ("item", "task"):
                    errors.append(
                        f"Step id '{step.id}' is reserved or used twice."
                    )
                seen_ids.add(step.id)
                phase_of[step.id] = index
                schema_of[step.id] = step.output_schema

        for index, phase in enumerate(plan.phases):
            for step in phase.steps:
                if step.agent not in known_agents:
                    errors.append(
                        f"Step '{step.id}' uses unknown agent '{step.agent}'."
                    )
                for ref in PLACEHOLDER.findall(step.prompt):
                    if ref == "item" and not step.for_each:
                        errors.append(
                            f"Step '{step.id}' uses {{item}} without for_each."
                        )
                    elif ref in phase_of and phase_of[ref] >= index:
                        errors.append(
                            f"Step '{step.id}' references '{ref}', which is not in an earlier phase."
                        )
                if step.for_each:
                    errors.extend(
                        self._check_for_each(
                            step, index, phase_of, schema_of
                        )
                    )

        for step_id in plan.return_steps:
            if step_id not in phase_of:
                errors.append(
                    f"return_steps names unknown step '{step_id}'."
                )
        if not plan.return_steps:
            errors.append("return_steps is empty.")

        static_count = sum(
            1
            for phase in plan.phases
            for step in phase.steps
            if not step.for_each
        )
        if static_count > self.max_agents:
            errors.append(
                f"The plan uses {static_count} agents; the limit is {self.max_agents}."
            )
        return errors

    @staticmethod
    def _check_for_each(
        step: Step,
        index: int,
        phase_of: Dict[str, int],
        schema_of: Dict[str, Optional[dict]],
    ) -> List[str]:
        """Check one for_each reference.

        Args:
            step: The fan-out step.
            index: Index of the step's phase.
            phase_of: Phase index of every step id.
            schema_of: Output schema of every step id.

        Returns:
            List[str]: The problems found.
        """
        source, _, field = step.for_each.partition(".")
        if not field:
            return [
                f"Step '{step.id}' for_each must look like '<step_id>.<field>'."
            ]
        if source not in phase_of or phase_of[source] >= index:
            return [
                f"Step '{step.id}' for_each source '{source}' is not in an earlier phase."
            ]
        properties = (schema_of[source] or {}).get("properties", {})
        if field not in properties:
            return [
                f"Step '{step.id}' for_each field '{field}' is not declared in the output_schema of '{source}'."
            ]
        if properties[field].get("type") not in (None, "array"):
            return [
                f"Step '{step.id}' for_each field '{source}.{field}' is not an array."
            ]
        return []

    def plan(
        self, task: str, img: Optional[str] = None
    ) -> WorkflowPlan:
        """Ask the lead for a valid plan, sending errors back for repair.

        Args:
            task: The task to plan for.
            img: Optional image passed to the lead.

        Returns:
            WorkflowPlan: A plan that passed validation.

        Raises:
            DynamicWorkflowError: If no valid plan is written within max_plan_attempts.
        """
        lead = self._lead
        prompt = self._planning_prompt(task)
        errors: List[str] = []
        for attempt in range(1, self.max_plan_attempts + 1):
            raw = (
                lead.run(prompt, img=img) if img else lead.run(prompt)
            )
            try:
                plan = WorkflowPlan.model_validate(_parse_json(raw))
                errors = self.validate_plan(plan)
            except (ValueError, ValidationError) as error:
                errors = [
                    f"The plan is not valid JSON for the schema: {error}"
                ]
            if not errors:
                return plan
            logger.warning(
                f"{self.name}: plan attempt {attempt} rejected: {errors}"
            )
            prompt = (
                "Your plan was rejected for these reasons:\n- "
                + "\n- ".join(errors)
                + "\n\nWrite a corrected plan.\n\n"
                + self._planning_prompt(task)
            )
        raise DynamicWorkflowError(
            f"No valid plan after {self.max_plan_attempts} attempt(s): {errors}"
        )

    def _render(
        self, prompt: str, outputs: Dict[str, Any], item: Any = None
    ) -> str:
        """Fill a step prompt's placeholders.

        Args:
            prompt: The step prompt.
            outputs: Outputs of the steps run so far.
            item: The fan-out item, if any.

        Returns:
            str: The rendered prompt.
        """

        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key == "item" and item is not None:
                return _as_text(item)
            if key in outputs:
                return _as_text(outputs[key])
            return match.group(0)

        return PLACEHOLDER.sub(replace, prompt)

    @staticmethod
    def _fan_out_items(step: Step, outputs: Dict[str, Any]) -> list:
        """Resolve the list a for_each step runs over.

        Args:
            step: The fan-out step.
            outputs: Outputs of the steps run so far.

        Returns:
            list: The items, flattened across a fan-out source's slots.

        Raises:
            DynamicWorkflowError: If the field is missing or not a list.
        """
        source, _, field = step.for_each.partition(".")
        value = outputs.get(source)
        slots = value if isinstance(value, list) else [value]
        items = []
        for slot in slots:
            if slot is None:
                continue
            if not isinstance(slot, dict) or not isinstance(
                slot.get(field), list
            ):
                raise DynamicWorkflowError(
                    f"Step '{step.id}' needs a list at '{step.for_each}', got {type(slot).__name__}."
                )
            items.extend(slot[field])
        return items

    def _run_agent(
        self, step: Step, prompt: str, plan: WorkflowPlan
    ) -> Tuple[Any, List[Agent]]:
        """Run one agent for a step, retrying on failure with a fresh agent.

        Args:
            step: The step being run.
            prompt: The rendered prompt.
            plan: The running plan.

        Returns:
            Tuple[Any, List[Agent]]: The output and every agent instance used.

        Raises:
            Exception: The last error once retries run out; it carries the agents used.
        """
        if step.output_schema:
            prompt += (
                "\n\nReturn ONLY JSON matching this schema, with no other text:\n"
                + json.dumps(step.output_schema)
            )
        used = []
        for attempt in range(self.max_retries + 1):
            agent = self._build_agent(step.agent, plan)
            used.append(agent)
            try:
                raw = agent.run(prompt)
                if step.output_schema:
                    return _parse_json(raw), used
                return raw, used
            except Exception as error:
                logger.warning(
                    f"{self.name}: step '{step.id}' attempt {attempt + 1} failed: {error}"
                )
                last_error = error
        last_error.agents_used = used
        raise last_error

    def _run_phase(
        self,
        phase: Phase,
        plan: WorkflowPlan,
        outputs: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        """Run every step of a phase concurrently and store their outputs.

        Args:
            phase: The phase to run.
            plan: The running plan.
            outputs: Outputs of the steps run so far; updated in place.
            state: Run bookkeeping; updated in place.

        Raises:
            DynamicWorkflowError: If a fan-out would pass max_agents or a single step fails.
        """
        jobs = []
        for step in phase.steps:
            if step.for_each:
                for slot, item in enumerate(
                    self._fan_out_items(step, outputs)
                ):
                    jobs.append((step, slot, item))
                outputs[step.id] = [None] * sum(
                    1 for job in jobs if job[0] is step
                )
            else:
                jobs.append((step, None, None))

        if state["agent_count"] + len(jobs) > self.max_agents:
            raise DynamicWorkflowError(
                f"Phase '{phase.title}' needs {len(jobs)} agents, which would pass "
                f"max_agents={self.max_agents} ({state['agent_count']} already run)."
            )
        state["agent_count"] += len(jobs)
        if not jobs:
            return

        snapshot = {"task": state["task"], **outputs}
        workers = min(self.max_concurrent_agents, len(jobs))
        with ContextThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                (
                    pool.submit(
                        self._run_agent,
                        step,
                        self._render(step.prompt, snapshot, item),
                        plan,
                    ),
                    step,
                    slot,
                    item,
                )
                for step, slot, item in jobs
            ]
            single_failures = []
            for future, step, slot, item in futures:
                try:
                    output, used = future.result()
                except Exception as error:
                    state["agents"].extend(
                        getattr(error, "agents_used", [])
                    )
                    state["failures"].append(
                        {
                            "phase": phase.title,
                            "step": step.id,
                            "item": item,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    if slot is None:
                        single_failures.append((step.id, error))
                    continue
                state["agents"].extend(used)
                if slot is None:
                    outputs[step.id] = output
                else:
                    outputs[step.id][slot] = output

        if single_failures:
            step_id, error = single_failures[0]
            raise DynamicWorkflowError(
                f"Step '{step_id}' in phase '{phase.title}' failed after "
                f"{self.max_retries + 1} attempt(s): {error}"
            ) from error

    def _combine_prompt(
        self, task: str, plan: WorkflowPlan, outputs: Dict[str, Any]
    ) -> str:
        """Prompt asking the lead to write the final answer.

        Args:
            task: The task.
            plan: The plan that ran.
            outputs: Outputs of every step.

        Returns:
            str: The combine prompt.
        """
        results = "\n\n".join(
            f"### {step_id}\n{_as_text(outputs.get(step_id))}"
            for step_id in plan.return_steps
        )
        return (
            f"Task:\n{task}\n\n"
            f"Plan that ran:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Outputs:\n{results}\n\n"
            "Write the final answer for the task from these outputs."
        )

    @property
    def usage(self) -> dict:
        """Provider token usage of the last run, lead included.

        Returns:
            dict: Summed token counts in the shape of Agent usage.
        """
        total = empty_usage()
        for agent in (self.last_run or {}).get("agents", []):
            for key, value in (
                getattr(agent, "usage", None) or {}
            ).items():
                if key in total:
                    total[key] += value
        return total

    @trace_run(
        "DynamicWorkflow.run",
        input_params=("task", "img"),
        usage=lambda workflow: workflow.usage,
    )
    def run(self, task: str, img: Optional[str] = None) -> Any:
        """Plan the task, run the plan's phases and return the combined result.

        Args:
            task: The task to run.
            img: Optional image passed to the lead when planning.

        Returns:
            Any: The conversation formatted by output_type.

        Raises:
            DynamicWorkflowError: If planning fails, a limit is passed, or a single step fails.
        """
        self._lead = self._fresh_lead()
        state: Dict[str, Any] = {
            "outputs": {},
            "failures": [],
            "task": task,
            "agent_count": 0,
            "agents": [self._lead],
        }
        self.last_plan = None
        self.last_run = state
        conversation = Conversation(name=self.name)
        conversation.add("user", task)

        plan = self.plan(task, img)
        self.last_plan = plan
        conversation.add(
            self._lead.agent_name, plan.model_dump_json()
        )

        outputs = state["outputs"]
        for phase in plan.phases:
            logger.info(f"{self.name}: running phase '{phase.title}'")
            self._run_phase(phase, plan, outputs, state)
            for step in phase.steps:
                conversation.add(
                    f"{phase.title} / {step.id}",
                    _as_text(outputs.get(step.id)),
                )

        answer = self._lead.run(
            self._combine_prompt(task, plan, outputs)
        )
        state["answer"] = answer
        conversation.add(self._lead.agent_name, answer)
        state["usage"] = self.usage
        return history_output_formatter(
            conversation, type=self.output_type
        )
