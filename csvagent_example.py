# Example usage
from swarms.structs.agent_loader import AgentLoader


# Example agent configurations
agent_configs = [
    {
        "agent_name": "Financial-Analysis-Agent",
        "system_prompt": "You are a financial expert...",
        "model_name": "gpt-4o",
        "max_loops": 1,
        "autosave": True,
        "dashboard": False,
        "verbose": True,
        "dynamic_temperature": True,
        "saved_state_path": "finance_agent.json",
        "user_name": "swarms_corp",
        "retry_attempts": 3,
        "context_length": 200000,
        "return_step_meta": False,
        "output_type": "string",
        "streaming": False,
    }
]


# Initialize loader
loader = AgentLoader("agents/my_agents")

# Create agents in CSV format
loader.create_agents(agent_configs, file_type="csv")

# Or create agents in JSON format
loader.create_agents(agent_configs, file_type="json")

# Load agents from either format
agents_from_csv = loader.load_agents(file_type="csv")
agents_from_json = loader.load_agents(file_type="json")
print(agents_from_csv)
