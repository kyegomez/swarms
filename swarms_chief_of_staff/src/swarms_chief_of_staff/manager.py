"""Chief of Staff: core task registry and orchestration helpers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid

from swarms.structs.agent import Agent
from swarms.tools.mcp_client_tools import get_mcp_tools_sync
from .storage import ChiefStorage


@dataclass
class Task:
    id: str
    title: str
    description: str
    status: str = "pending"
    result: Optional[Any] = None
    subtasks: List[str] = field(default_factory=list)


class ChiefOfStaff:
    def __init__(self, db_path: str = None):
        self.tasks: Dict[str, Task] = {}
        self.storage = ChiefStorage(db_path) if db_path is not None else None
        if self.storage:
            # load persisted tasks into memory
            for t in self.storage.load_all_tasks():
                self.tasks[t.id] = t

    def create_task(self, title: str, description: str) -> Task:
        tid = f"task-{uuid.uuid4().hex}"
        t = Task(id=tid, title=title, description=description)
        self.tasks[tid] = t
        if self.storage:
            self.storage.save_task(t)
        return t

    def list_tasks(self) -> List[Task]:
        return list(self.tasks.values())

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def plan_task(self, task_id: str, model_name: str = "gpt-4.1") -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise KeyError("task not found")

        system_prompt = (
            "You are a research assistant. Break down the following task into a set of actionable subtasks and suggested tools. "
            "Respond as JSON with keys: goals (list), subtasks (list of {id,title,description}), suggestions (str)."
        )
        agent = Agent(agent_name="ChiefResearcher", system_prompt=system_prompt, model_name=model_name, max_loops=1)
        raw = agent.run(task.description)
        try:
            import json

            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}

        task.result = parsed
        task.status = "planned"
        if self.storage:
            self.storage.update_task(task)
        return parsed

    def spawn_subagent(self, task_id: str, subtask_spec: Dict[str, Any], model_name: str = "gpt-4.1") -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise KeyError("task not found")

        name = subtask_spec.get("title", "subtask")
        prompt = subtask_spec.get("description", "Please complete the subtask")

        mcp_url = subtask_spec.get("mcp_url")
        if mcp_url:
            try:
                tools = get_mcp_tools_sync(server_path=mcp_url, format="openai")
            except Exception:
                tools = None
        else:
            tools = None
        # If MCP tools were provided (OpenAI function schemas), pass them to agent via tools_list_dictionary
        agent_kwargs = {}
        if tools:
            agent_kwargs["tools_list_dictionary"] = tools

        agent = Agent(agent_name=f"worker-{name}", system_prompt=prompt, model_name=model_name, max_loops=1, **agent_kwargs)
        output = agent.run(prompt)
        sub_id = f"sub-{uuid.uuid4().hex}"
        task.subtasks.append(sub_id)
        if self.storage:
            self.storage.update_task(task)
        return {"subtask_id": sub_id, "output": output}

    def execute_plan(self, task_id: str, model_name: str = "gpt-4.1") -> Dict[str, Any]:
        """Execute all subtasks from a planned task concurrently and collect outputs.

        Expects task.result to contain a 'subtasks' list of dicts with 'title' and 'description'.
        """
        import concurrent.futures

        task = self.get_task(task_id)
        if not task:
            raise KeyError("task not found")
        if not task.result or not isinstance(task.result, dict):
            raise ValueError("task has no plan to execute")

        subtasks = task.result.get("subtasks") or []
        results = []

        def _run_sub(st):
            try:
                return self.spawn_subagent(task_id, st, model_name=model_name)
            except Exception as e:
                return {"subtask_id": None, "error": str(e)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(subtasks)))) as ex:
            futures = {ex.submit(_run_sub, st): st for st in subtasks}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        task.result.setdefault("execution", {})
        task.result["execution"]["outputs"] = results
        task.status = "executed"
        if self.storage:
            self.storage.update_task(task)
        return {"outputs": results}

    def approve_plan(self, task_id: str) -> None:
        t = self.get_task(task_id)
        if not t:
            raise KeyError("task not found")
        if t.status != "planned":
            raise ValueError("task is not in planned state")
        t.status = "approved"
        if self.storage:
            self.storage.update_task(t)
