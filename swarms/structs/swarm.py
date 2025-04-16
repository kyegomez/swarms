from ..utils.language_config import language_config, Language

class Swarm:
    def __init__(
        self,
        agents: List[Agent],
        language: Language = Language.ENGLISH,
        **kwargs
    ):
        self.agents = agents
        self.language = language
        language_config.set_language(language)

    def run(self, task: str) -> Dict[str, Any]:
        """Run the swarm with the given task."""
        try:
            print(language_config.get_translation("status_messages", "task_started"))
            results = {}
            
            for agent in self.agents:
                print(f"{agent.name}: {language_config.get_translation('status_messages', 'task_in_progress')}")
                results[agent.name] = agent.run(task)
            
            print(language_config.get_translation("status_messages", "task_completed"))
            return results
        except Exception as e:
            error_msg = language_config.get_translation("error_messages", "task_failed")
            print(f"{error_msg}: {str(e)}")
            raise

    def add_agent(self, agent: Agent) -> None:
        """Add a new agent to the swarm."""
        if agent.name in [a.name for a in self.agents]:
            error_msg = language_config.get_translation("error_messages", "agent_not_found")
            raise ValueError(f"{error_msg}: {agent.name}")
        self.agents.append(agent) 