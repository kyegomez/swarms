"""
SkillOrchestra: Skill-Aware Agent Orchestration

Based on the paper "SkillOrchestra: Learning to Route Agents via Skill Transfer"
(https://arxiv.org/abs/2602.19672).

Instead of end-to-end RL routing, this class maintains a Skill Handbook that profiles
each agent on fine-grained skills, infers which skills a task requires via LLM, and
matches agents to tasks via explicit competence-cost scoring.

Pipeline:
1. Task → Skill Inference (LLM identifies required skills)
2. Agent Scoring (pure math: weighted competence-cost)
3. Agent Selection (top-k)
4. Execution
5. Learning (optional: LLM evaluates output, updates profiles via EMA)
"""

import json
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import list_all_agents
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM
from swarms.utils.output_types import OutputType
from swarms.utils.swarm_autosave import get_swarm_workspace_dir


# ─── Pydantic Models ─────────────────────────────────────────────


class SkillDefinition(BaseModel):
    """A single fine-grained skill that agents can possess."""

    name: str = Field(
        description="Short, unique name for the skill (e.g., 'code_review', 'financial_analysis')"
    )
    description: str = Field(
        description="Detailed description of what this skill entails"
    )
    category: Optional[str] = Field(
        None,
        description="Optional category grouping (e.g., 'engineering', 'analysis')",
    )


class AgentSkillProfile(BaseModel):
    """An agent's competence on a specific skill, with cost estimate."""

    skill_name: str = Field(
        description="Name of the skill being profiled"
    )
    competence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated probability of successful execution (0-1)",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Estimated relative cost for this skill",
    )
    execution_count: int = Field(
        default=0,
        description="Number of times this agent has executed this skill",
    )
    success_count: int = Field(
        default=0,
        description="Number of successful executions",
    )


class AgentProfile(BaseModel):
    """Complete skill profile for a single agent."""

    agent_name: str = Field(description="Name of the agent")
    skill_profiles: List[AgentSkillProfile] = Field(
        default_factory=list,
        description="Skill profiles for this agent",
    )


class SkillHandbook(BaseModel):
    """Central data structure mapping skills to agent competence and cost."""

    skills: List[SkillDefinition] = Field(
        default_factory=list,
        description="All defined skills",
    )
    agent_profiles: List[AgentProfile] = Field(
        default_factory=list,
        description="Per-agent skill profiles",
    )


class InferredSkill(BaseModel):
    """A skill inferred as required for a task."""

    skill_name: str = Field(
        description="Name of the required skill (must match handbook)"
    )
    importance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How important this skill is for the task (0-1)",
    )
    reasoning: str = Field(description="Why this skill is needed")


class TaskSkillInference(BaseModel):
    """LLM output: skills required by a given task."""

    task_summary: str = Field(
        description="Brief summary of the task's core requirements"
    )
    required_skills: List[InferredSkill] = Field(
        description="Skills required to complete this task"
    )


class AgentSelectionResult(BaseModel):
    """Result of agent selection scoring."""

    agent_name: str = Field(description="Name of the selected agent")
    score: float = Field(
        description="Composite competence-cost score"
    )
    reasoning: str = Field(description="Why this agent was selected")
    assigned_task: Optional[str] = Field(
        None,
        description="Optionally rephrased task for this agent",
    )


class ExecutionFeedback(BaseModel):
    """Post-execution feedback for updating skill profiles."""

    agent_name: str = Field(
        description="Name of the agent that executed"
    )
    skills_used: List[str] = Field(
        description="Skills that were exercised"
    )
    success: bool = Field(
        description="Whether execution was successful"
    )
    quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quality score of the output (0-1)",
    )
    reasoning: str = Field(
        description="Explanation of the quality assessment"
    )


# ─── Prompts ─────────────────────────────────────────────────────

SKILL_HANDBOOK_GENERATION_PROMPT = """You are an expert at analyzing agent capabilities and identifying fine-grained, practical skills.

Given a list of agents with their names and descriptions, you must:
1. Identify a comprehensive set of fine-grained, PRACTICAL skills that these agents can perform
2. For each agent, estimate their competence (0-1) on each skill based on their description
3. For each agent-skill pair, estimate relative cost (1.0 = average, <1 = cheaper, >1 = more expensive)

IMPORTANT: Skills must be concrete and task-oriented. Good examples: "python_coding", "data_analysis", "api_design", "code_review", "technical_writing", "sql_queries", "debugging", "creative_writing".
Bad examples: "autonomous_task_execution", "swarm_intelligence", "experience_learning" — these are too abstract.

Focus on what each agent can actually DO based on their name and description. Ignore any framework/infrastructure details.

Available agents:
{agent_descriptions}

Respond with the complete skill handbook as JSON matching the required schema."""


SKILL_INFERENCE_PROMPT = """You are a task analysis specialist. Given a task and available skills, determine which skills are required.

Available skills:
{skill_list}

For each required skill, assign an importance weight (0-1) and explain why it's needed.
Only include skills from the available list. Do not invent new skill names."""


EXECUTION_EVALUATION_PROMPT = """You are evaluating the quality of an agent's task execution.

Task: {task}
Agent: {agent_name}
Skills exercised: {skills_used}
Agent output:
{agent_output}

Evaluate:
1. Was the execution successful?
2. Quality score (0-1)
3. Which skills were demonstrated in the output?
4. Brief reasoning."""


# ─── SkillOrchestra Class ────────────────────────────────────────


class SkillOrchestra:
    """
    Skill-aware agent orchestration based on the SkillOrchestra paper.

    Routes tasks to agents by maintaining a Skill Handbook (skill definitions +
    per-agent competence/cost profiles), using an LLM to infer task skill requirements,
    scoring agents via weighted competence-cost matching, and optionally learning
    from execution feedback.

    Args:
        name: Name of the orchestrator.
        description: Description of the orchestrator.
        agents: List of agents to orchestrate.
        max_loops: Maximum execution-feedback loops per task.
        output_type: Output format type.
        model: LLM model name for skill inference and evaluation.
        temperature: LLM temperature.
        skill_handbook: Optional pre-built skill handbook. If None, auto-generated.
        auto_generate_skills: Whether to auto-generate handbook from agent descriptions.
        cost_weight: Weight for cost component in scoring (0-1).
        competence_weight: Weight for competence component in scoring (0-1).
        top_k_agents: Number of agents to select per task.
        learning_enabled: Whether to update skill profiles after execution.
        learning_rate: EMA learning rate for profile updates.
        autosave: Whether to save conversation history.
        verbose: Whether to log detailed information.
        print_on: Whether to print panels to console.

    Example:
        >>> from swarms import Agent, SkillOrchestra
        >>> agents = [
        ...     Agent(agent_name="CodeExpert", description="Expert Python developer"),
        ...     Agent(agent_name="Writer", description="Technical writing specialist"),
        ...     Agent(agent_name="Researcher", description="Research and analysis expert"),
        ... ]
        >>> orchestra = SkillOrchestra(agents=agents)
        >>> result = orchestra.run("Write a Python function to parse CSV files")
    """

    def __init__(
        self,
        name: str = "SkillOrchestra",
        description: str = "Skill-aware agent orchestration via skill profiling and matching",
        agents: List[Union[Agent, Callable]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        skill_handbook: Optional[SkillHandbook] = None,
        auto_generate_skills: bool = True,
        cost_weight: float = 0.3,
        competence_weight: float = 0.7,
        top_k_agents: int = 1,
        learning_enabled: bool = True,
        learning_rate: float = 0.1,
        autosave: bool = True,
        verbose: bool = False,
        print_on: bool = True,
        *args,
        **kwargs,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.agents = agents or []
        self.max_loops = max_loops
        self.output_type = output_type
        self.model = model
        self.temperature = temperature
        self.cost_weight = cost_weight
        self.competence_weight = competence_weight
        self.top_k_agents = top_k_agents
        self.learning_enabled = learning_enabled
        self.learning_rate = learning_rate
        self.autosave = autosave
        self.verbose = verbose
        self.print_on = print_on
        self.swarm_workspace_dir = None

        self._reliability_check()

        # Build agent lookup map
        self.agent_map: Dict[str, Any] = {}
        for agent in self.agents:
            agent_name = getattr(
                agent,
                "agent_name",
                getattr(agent, "name", str(agent)),
            )
            self.agent_map[agent_name] = agent

        # Conversation tracking
        self.conversation = Conversation(time_enabled=False)

        list_all_agents(
            agents=self.agents,
            conversation=self.conversation,
            name=self.name,
            description=self.description,
            add_to_conversation=True,
        )

        # Skill handbook
        if skill_handbook is not None:
            self.skill_handbook = skill_handbook
        elif auto_generate_skills:
            self.skill_handbook = self._auto_generate_handbook()
        else:
            self.skill_handbook = SkillHandbook()

        # Autosave setup
        if self.autosave:
            self._setup_autosave()

        logger.info(
            f"SkillOrchestra '{self.name}' initialized with "
            f"{len(self.agents)} agents, "
            f"{len(self.skill_handbook.skills)} skills"
        )

    def _reliability_check(self):
        """Validate configuration."""
        if not self.agents or len(self.agents) == 0:
            raise ValueError(
                "SkillOrchestra requires at least one agent"
            )

        if self.cost_weight + self.competence_weight == 0:
            raise ValueError(
                "cost_weight and competence_weight cannot both be zero"
            )

        if self.top_k_agents < 1:
            raise ValueError("top_k_agents must be at least 1")

        if self.max_loops < 1:
            raise ValueError("max_loops must be at least 1")

    def _get_agent_descriptions(self) -> str:
        """Build a formatted string of agent names and descriptions."""
        lines = []
        for agent in self.agents:
            agent_name = getattr(
                agent,
                "agent_name",
                getattr(agent, "name", "Unknown"),
            )
            desc = getattr(agent, "description", None)
            if not desc:
                desc = getattr(agent, "system_prompt", "")
                if len(desc) > 200:
                    desc = desc[:200] + "..."
            lines.append(f"- {agent_name}: {desc}")
        return "\n".join(lines)

    def _auto_generate_handbook(self) -> SkillHandbook:
        """Use LLM to generate initial skill handbook from agent descriptions."""
        logger.info(
            "Auto-generating skill handbook from agent descriptions"
        )

        agent_descriptions = self._get_agent_descriptions()
        prompt = SKILL_HANDBOOK_GENERATION_PROMPT.format(
            agent_descriptions=agent_descriptions
        )

        try:
            llm = LiteLLM(
                model_name=self.model,
                temperature=self.temperature,
                response_format=SkillHandbook,
            )
            response = llm.run(prompt)
            handbook = SkillHandbook.model_validate_json(response)

            logger.info(
                f"Generated handbook with {len(handbook.skills)} skills "
                f"and {len(handbook.agent_profiles)} agent profiles"
            )
            return handbook

        except Exception as e:
            logger.warning(
                f"Failed to auto-generate handbook: {e}. "
                "Using default handbook."
            )
            return self._build_default_handbook()

    def _build_default_handbook(self) -> SkillHandbook:
        """Build a minimal default handbook when auto-generation fails."""
        skills = [
            SkillDefinition(
                name="general_task_execution",
                description="General ability to understand and execute tasks",
            )
        ]

        agent_profiles = []
        for agent in self.agents:
            agent_name = getattr(
                agent,
                "agent_name",
                getattr(agent, "name", "Unknown"),
            )
            agent_profiles.append(
                AgentProfile(
                    agent_name=agent_name,
                    skill_profiles=[
                        AgentSkillProfile(
                            skill_name="general_task_execution",
                            competence=0.5,
                            cost=1.0,
                        )
                    ],
                )
            )

        return SkillHandbook(
            skills=skills, agent_profiles=agent_profiles
        )

    def _build_skill_list_str(self) -> str:
        """Build a formatted string of available skills for prompts."""
        lines = []
        for skill in self.skill_handbook.skills:
            cat = f" [{skill.category}]" if skill.category else ""
            lines.append(f"- {skill.name}{cat}: {skill.description}")
        return "\n".join(lines)

    def _infer_task_skills(self, task: str) -> TaskSkillInference:
        """Use LLM to infer which skills a task requires."""
        skill_list = self._build_skill_list_str()

        system_prompt = SKILL_INFERENCE_PROMPT.format(
            skill_list=skill_list
        )

        llm = LiteLLM(
            model_name=self.model,
            system_prompt=system_prompt,
            temperature=self.temperature,
            response_format=TaskSkillInference,
        )

        response = llm.run(task)
        inference = TaskSkillInference.model_validate_json(response)

        # Filter out skills not in the handbook
        valid_skill_names = {
            s.name for s in self.skill_handbook.skills
        }
        inference.required_skills = [
            s
            for s in inference.required_skills
            if s.skill_name in valid_skill_names
        ]

        if self.verbose:
            logger.info(
                f"Inferred {len(inference.required_skills)} required skills "
                f"for task: {inference.task_summary}"
            )

        return inference

    def _score_agents(
        self, skill_inference: TaskSkillInference
    ) -> List[AgentSelectionResult]:
        """Score agents based on skill match with competence-cost weighting."""
        required_skills = skill_inference.required_skills
        total_importance = sum(s.importance for s in required_skills)

        if total_importance == 0:
            # No skills inferred — return all agents with equal score
            return [
                AgentSelectionResult(
                    agent_name=ap.agent_name,
                    score=0.5,
                    reasoning="No specific skills inferred; default score",
                )
                for ap in self.skill_handbook.agent_profiles
            ]

        # Pre-compute cost normalization ranges per skill
        cost_ranges: Dict[str, tuple] = {}
        for skill in required_skills:
            costs = []
            for ap in self.skill_handbook.agent_profiles:
                for sp in ap.skill_profiles:
                    if sp.skill_name == skill.skill_name:
                        costs.append(sp.cost)
            if costs:
                cost_ranges[skill.skill_name] = (
                    min(costs),
                    max(costs),
                )
            else:
                cost_ranges[skill.skill_name] = (1.0, 1.0)

        results = []
        for agent_profile in self.skill_handbook.agent_profiles:
            skill_map = {
                sp.skill_name: sp
                for sp in agent_profile.skill_profiles
            }

            score = 0.0
            matched_skills = []

            for req_skill in required_skills:
                sp = skill_map.get(req_skill.skill_name)
                if sp is not None:
                    # Competence component
                    competence_score = (
                        sp.competence * req_skill.importance
                    )

                    # Cost component (lower cost = higher score)
                    min_c, max_c = cost_ranges[req_skill.skill_name]
                    if max_c > min_c:
                        normalized_cost = 1.0 - (
                            (sp.cost - min_c) / (max_c - min_c)
                        )
                    else:
                        normalized_cost = 1.0
                    cost_score = (
                        normalized_cost * req_skill.importance
                    )

                    score += (
                        self.competence_weight * competence_score
                        + self.cost_weight * cost_score
                    )
                    matched_skills.append(req_skill.skill_name)

            final_score = score / total_importance

            reasoning = f"Matched {len(matched_skills)}/{len(required_skills)} skills"
            if matched_skills:
                reasoning += f": {', '.join(matched_skills)}"

            results.append(
                AgentSelectionResult(
                    agent_name=agent_profile.agent_name,
                    score=round(final_score, 4),
                    reasoning=reasoning,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _select_agents(
        self, scored_agents: List[AgentSelectionResult]
    ) -> List[AgentSelectionResult]:
        """Select top-k agents from scored list."""
        return scored_agents[: self.top_k_agents]

    def _execute_agents(
        self,
        selected: List[AgentSelectionResult],
        task: str,
        img: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run selected agents on the task."""
        results = []

        if len(selected) == 1:
            # Single agent — run directly
            sel = selected[0]
            agent = self.agent_map.get(sel.agent_name)
            if agent is None:
                logger.error(
                    f"Agent '{sel.agent_name}' not found in agent map"
                )
                return results

            agent_task = sel.assigned_task or task

            if self.verbose:
                logger.info(
                    f"Executing agent '{sel.agent_name}' on task"
                )

            response = agent.run(task=agent_task, img=img)

            self.conversation.add(
                role=sel.agent_name, content=str(response)
            )
            results.append(
                {
                    "agent_name": sel.agent_name,
                    "response": response,
                }
            )

        else:
            # Multiple agents — run concurrently
            with ThreadPoolExecutor(
                max_workers=min(len(selected), os.cpu_count() or 4)
            ) as executor:
                future_to_agent = {}
                for sel in selected:
                    agent = self.agent_map.get(sel.agent_name)
                    if agent is None:
                        logger.error(
                            f"Agent '{sel.agent_name}' not found"
                        )
                        continue
                    agent_task = sel.assigned_task or task
                    future = executor.submit(
                        agent.run, task=agent_task, img=img
                    )
                    future_to_agent[future] = sel.agent_name

                for future in as_completed(future_to_agent):
                    agent_name = future_to_agent[future]
                    try:
                        response = future.result()
                        self.conversation.add(
                            role=agent_name,
                            content=str(response),
                        )
                        results.append(
                            {
                                "agent_name": agent_name,
                                "response": response,
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Agent '{agent_name}' failed: {e}"
                        )
                        results.append(
                            {
                                "agent_name": agent_name,
                                "response": f"Error: {e}",
                            }
                        )

        return results

    def _evaluate_and_update(
        self,
        task: str,
        results: List[Dict[str, Any]],
        skill_inference: TaskSkillInference,
    ) -> None:
        """Evaluate execution quality and update skill profiles via EMA."""
        skills_used = [
            s.skill_name for s in skill_inference.required_skills
        ]

        for result in results:
            agent_name = result["agent_name"]
            agent_output = str(result["response"])

            # Truncate long outputs to save tokens
            if len(agent_output) > 2000:
                agent_output = agent_output[:2000] + "...[truncated]"

            prompt = EXECUTION_EVALUATION_PROMPT.format(
                task=task,
                agent_name=agent_name,
                skills_used=", ".join(skills_used),
                agent_output=agent_output,
            )

            try:
                llm = LiteLLM(
                    model_name=self.model,
                    temperature=self.temperature,
                    response_format=ExecutionFeedback,
                )
                response = llm.run(prompt)
                feedback = ExecutionFeedback.model_validate_json(
                    response
                )

                self._update_profiles(feedback)

                if self.verbose:
                    logger.info(
                        f"Updated profiles for '{agent_name}': "
                        f"success={feedback.success}, "
                        f"quality={feedback.quality_score:.2f}"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to evaluate agent '{agent_name}': {e}"
                )

    def _update_profiles(self, feedback: ExecutionFeedback) -> None:
        """Update agent skill profiles using exponential moving average."""
        for ap in self.skill_handbook.agent_profiles:
            if ap.agent_name != feedback.agent_name:
                continue

            skill_map = {
                sp.skill_name: sp for sp in ap.skill_profiles
            }

            for skill_name in feedback.skills_used:
                sp = skill_map.get(skill_name)
                if sp is not None:
                    # EMA update
                    sp.competence = (
                        sp.competence * (1 - self.learning_rate)
                        + feedback.quality_score * self.learning_rate
                    )
                    sp.execution_count += 1
                    if feedback.success:
                        sp.success_count += 1

            break

    def _setup_autosave(self):
        """Set up autosave workspace directory."""
        try:
            self.swarm_workspace_dir = get_swarm_workspace_dir(
                class_name="SkillOrchestra",
                swarm_name=self.name,
            )
        except Exception as e:
            logger.warning(
                f"Failed to set up autosave workspace: {e}"
            )
            self.swarm_workspace_dir = None

    def _save_conversation_history(self):
        """Save conversation history and handbook to workspace."""
        if not self.swarm_workspace_dir:
            return

        try:
            # Save conversation
            conv_path = os.path.join(
                self.swarm_workspace_dir, "conversation.json"
            )
            with open(conv_path, "w") as f:
                json.dump(
                    self.conversation.to_dict(),
                    f,
                    indent=2,
                    default=str,
                )

            # Save skill handbook
            handbook_path = os.path.join(
                self.swarm_workspace_dir, "skill_handbook.json"
            )
            with open(handbook_path, "w") as f:
                f.write(self.skill_handbook.model_dump_json(indent=2))

        except Exception as e:
            logger.warning(
                f"Failed to save conversation history: {e}"
            )

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Run the SkillOrchestra pipeline on a task.

        Steps:
        1. Infer required skills via LLM
        2. Score agents on skill match (pure math)
        3. Select top-k agents
        4. Execute selected agents
        5. Optionally learn from results

        Args:
            task: The task to execute.
            img: Optional image input.
            imgs: Optional list of image inputs.

        Returns:
            Formatted output based on output_type.
        """
        try:
            self.conversation.add(role="User", content=task)
            results = []

            for loop_idx in range(self.max_loops):
                current_task = (
                    task
                    if loop_idx == 0
                    else (
                        f"Previous results:\n{json.dumps(results, default=str)}\n\n"
                        f"Original task: {task}\n\nRefine and improve the response."
                    )
                )

                # Step 1: Infer required skills
                skill_inference = self._infer_task_skills(
                    current_task
                )

                if self.print_on:
                    inference_info = (
                        f"Task: {skill_inference.task_summary}\n"
                        f"Required skills:\n"
                    )
                    for s in skill_inference.required_skills:
                        inference_info += (
                            f"  - {s.skill_name} "
                            f"(importance: {s.importance:.2f}): "
                            f"{s.reasoning}\n"
                        )
                    formatter.print_panel(
                        inference_info,
                        title=f"{self.name} - Skill Inference",
                    )

                self.conversation.add(
                    role=self.name,
                    content=f"Skill Inference: {skill_inference.model_dump_json()}",
                )

                # Step 2: Score agents
                scored_agents = self._score_agents(skill_inference)

                # Step 3: Select top-k
                selected = self._select_agents(scored_agents)

                if self.print_on:
                    selection_info = "\n".join(
                        f"  {a.agent_name}: score={a.score:.4f} - {a.reasoning}"
                        for a in selected
                    )
                    formatter.print_panel(
                        selection_info,
                        title=f"{self.name} - Agent Selection (loop {loop_idx + 1}/{self.max_loops})",
                    )

                self.conversation.add(
                    role=self.name,
                    content=f"Selected agents: {[a.agent_name for a in selected]}",
                )

                # Step 4: Execute
                results = self._execute_agents(
                    selected, current_task, img=img, **kwargs
                )

                # Step 5: Learn
                if self.learning_enabled:
                    self._evaluate_and_update(
                        current_task, results, skill_inference
                    )

            # Autosave
            if self.autosave:
                self._save_conversation_history()

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )

        except Exception as e:
            logger.error(
                f"Error in SkillOrchestra: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def __call__(self, task: str, *args, **kwargs) -> Any:
        """Callable interface — delegates to run()."""
        return self.run(task, *args, **kwargs)

    def batch_run(self, tasks: List[str]) -> List[Any]:
        """Run multiple tasks sequentially."""
        return [self.run(task) for task in tasks]

    def concurrent_batch_run(self, tasks: List[str]) -> List[Any]:
        """Run multiple tasks concurrently."""
        results = []
        with ThreadPoolExecutor(
            max_workers=min(len(tasks), os.cpu_count() or 4)
        ) as executor:
            futures = {
                executor.submit(self.run, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    results.append(f"Error: {e}")
        return results

    def get_handbook(self) -> dict:
        """Return the current skill handbook as a dictionary."""
        return self.skill_handbook.model_dump()

    def update_handbook(self, handbook: SkillHandbook) -> None:
        """Replace the skill handbook."""
        self.skill_handbook = handbook
        logger.info(
            f"Updated skill handbook: {len(handbook.skills)} skills, "
            f"{len(handbook.agent_profiles)} agent profiles"
        )
