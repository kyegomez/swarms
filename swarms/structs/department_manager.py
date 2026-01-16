"""DepartmentManager: manage named departments of agents."""
from typing import Any, Optional


class DepartmentManager:
    """Manage departments (list of agent lists) for hierarchical swarms.

    Departments are named groups of agents. This manager provides helpers to
    add agents, list departments, and flatten agents for swarm execution.
    """

    def __init__(self):
        self.departments: dict[str, list[Any]] = {}

    def add_department(self, name: str, agents: Optional[list[Any]] = None):
        self.departments[name] = agents or []

    def add_agent_to_department(self, dept_name: str, agent: Any):
        if dept_name not in self.departments:
            self.add_department(dept_name)
        self.departments[dept_name].append(agent)

    def list_departments(self) -> list[str]:
        return list(self.departments.keys())

    def flatten_agents(self) -> list[Any]:
        out: list[Any] = []
        for agents in self.departments.values():
            out.extend(agents)
        return out
