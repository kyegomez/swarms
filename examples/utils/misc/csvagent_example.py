# Example usage
from swarms import AgentLoader
from pydantic import ValidationError


if __name__ == "__main__":
    # Example agent configurations
    agent_configs = [
        {
            "agent_name": "Financial-Analysis-Agent",
            "system_prompt": "You are a financial expert...",
            "model_name": "gpt-5.4",  # Updated to correct model name
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
        loader = AgentLoader()

        # Create CSV with initial agents
        loader.create_agent_file("agents.csv", agent_configs)

        # Load agents from CSV
        agents = loader.load_agents_from_csv("agents.csv")

        # Use an agent
        if agents:
            financial_agent = agents[0]
            financial_agent.run(
                "How can I establish a ROTH IRA to buy stocks and get a tax break?"
            )

    except ValidationError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
