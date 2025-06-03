# Example usage
from pathlib import Path
from swarms.structs.csv_to_agent import (
    AgentLoader,
    AgentValidationError,
)


if __name__ == "__main__":
    # Example agent configurations
    agent_configs = [
        {
            "agent_name": "Financial-Analysis-Agent",
            "system_prompt": "You are a financial expert...",
            "model_name": "gpt-4o-mini",  # Updated to correct model name
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

    try:
        # Initialize CSV manager
        csv_manager = AgentLoader(Path("agents.csv"))

        # Create CSV with initial agents
        csv_manager.create_agent_csv(agent_configs)

        # Load agents from CSV
        agents = csv_manager.load_agents()

        # Use an agent
        if agents:
            financial_agent = agents[0]
            financial_agent.run(
                "How can I establish a ROTH IRA to buy stocks and get a tax break?"
            )

    except AgentValidationError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
