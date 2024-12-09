from typing import List
from .notification_manager import NotificationManager, UpdateMetadata
from .agent import Agent

class Orchestrator:
    def __init__(self):
        self.notification_manager = NotificationManager()
        self.agents: List[Agent] = []
        
    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator"""
        self.agents.append(agent)
        self.notification_manager.register_agent(agent.notification_profile)
        
    def handle_vector_db_update(self, update_metadata: UpdateMetadata):
        """Handle a vector DB update and notify relevant agents"""
        # Get list of agents to notify
        agents_to_notify = self.notification_manager.get_agents_to_notify(
            update_metadata
        )
        
        # Notify relevant agents
        for agent in self.agents:
            if agent.agent_name in agents_to_notify:
                agent.handle_vector_db_update(update_metadata)
                
    def update_agent_task_context(self, agent_name: str, task_context: str):
        """Update an agent's current task context"""
        if agent_name in self.notification_manager.agent_profiles:
            self.notification_manager.agent_profiles[
                agent_name
            ].current_task_context = task_context 