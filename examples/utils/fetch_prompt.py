from dotenv import load_dotenv

from swarms.agents.agent_marketplace_handler import (
    AgentMarketplaceHandler,
)

load_dotenv()

if __name__ == "__main__":
    prompt = AgentMarketplaceHandler.fetch(
        prompt_id="0ff9cc2f-390a-4eb1-9d3d-3a045cd2682e"
    )
    print(prompt)
