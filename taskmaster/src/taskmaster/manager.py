"""Core TaskManager and task definitions.

This provides an in-memory task registry, simple planning via a ResearchAgent wrapper,
and spawning of sub-agents using the `swarms` package. Implementations are intentionally
minimal so they can be extended for production use.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid

from swarms.structs.agent import Agent
from swarms.tools.mcp_client_tools import get_mcp_tools_sync


@dataclass
class Task:
    id: str
    title: str
    description: str
    status: str = "pending"
    result: Optional[Any] = None
    subtasks: List[str] = field(default_factory=list)


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def create_task(self, title: str, description: str) -> Task:
        tid = f"task-{uuid.uuid4().hex}"
        t = Task(id=tid, title=title, description=description)
        self.tasks[tid] = t
        return t

    def list_tasks(self) -> List[Task]:
        return list(self.tasks.values())

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def plan_task(self, task_id: str, model_name: str = "gpt-4.1") -> Dict[str, Any]:
        """Run a lightweight research agent to propose a plan for the task.
        Uses `swarms.structs.agent.Agent` with a small system prompt.
        Returns proposed plan and stores in task.result (pending human approval).
        """
        task = self.get_task(task_id)
        if not task:
            raise KeyError("task not found")

        system_prompt = (
            "You are a research assistant. Break down the following task into a set of actionable subtasks and suggested tools. "
            "Respond as JSON with keys: goals (list), subtasks (list of {id,title,description}), suggestions (str)."
        )
        agent = Agent(agent_name="TaskResearcher", system_prompt=system_prompt, model_name=model_name, max_loops=1)
        raw = agent.run(task.description)
        # best-effort parse: Agent may return text. Try to return raw string if parse fails.
        try:
            import json

            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}

        task.result = parsed
        task.status = "planned"
        return parsed

    def spawn_subagent(self, task_id: str, subtask_spec: Dict[str, Any], model_name: str = "gpt-4.1") -> Dict[str, Any]:
        """Spawn a sub-agent to execute a subtask. Returns the agent output.
        Uses MCP tools if provided in subtask_spec['mcp_url'] to extend capabilities.
        """
        task = self.get_task(task_id)
        if not task:
            raise KeyError("task not found")

        name = subtask_spec.get("title", "subtask")
        prompt = subtask_spec.get("description", "Please complete the subtask")

        # Optionally load MCP tools
        mcp_url = subtask_spec.get("mcp_url")
        if mcp_url:
            try:
                tools = get_mcp_tools_sync(server_path=mcp_url, format="openai")
            except Exception:
                tools = None
        else:
            tools = None

        agent = Agent(agent_name=f"worker-{name}", system_prompt=prompt, model_name=model_name, max_loops=1)
        output = agent.run(prompt)
        # record subtask id
        sub_id = f"sub-{uuid.uuid4().hex}"
        task.subtasks.append(sub_id)
        return {"subtask_id": sub_id, "output": output}

    def approve_plan(self, task_id: str) -> None:
        t = self.get_task(task_id)
        if not t:
            raise KeyError("task not found")
        if t.status != "planned":
            raise ValueError("task is not in planned state")
        t.status = "approved"

