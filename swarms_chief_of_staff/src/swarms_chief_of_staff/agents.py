from typing import Optional
from swarms.structs.agent import Agent


def create_research_agent(name: str = "ResearchAgent", model: str = "gpt-4.1", system_prompt: Optional[str] = None) -> Agent:
    if system_prompt is None:
        system_prompt = (
            "You are a research assistant. Produce structured, actionable plans and concise summaries. Reply in JSON when asked."
        )
    return Agent(agent_name=name, system_prompt=system_prompt, model_name=model, max_loops=1)


def create_mcp_enabled_agent(name: str, mcp_url: str, model: str = "gpt-4.1") -> Agent:
    system_prompt = f"You are an agent that can call remote tools via MCP at {mcp_url}. Use tools when needed."
    return Agent(agent_name=name, system_prompt=system_prompt, model_name=model, max_loops=1)
