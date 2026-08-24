from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AgentSpec(BaseModel):
    agent_name: Optional[str] = Field(
        # default=None,
        description="The unique name assigned to the agent, which identifies its role and functionality within the swarm.",
    )
    description: Optional[str] = Field(
        default=None,
        description="A detailed explanation of the agent's purpose, capabilities, and any specific tasks it is designed to perform.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="The initial instruction or context provided to the agent, guiding its behavior and responses during execution.",
    )
    model_name: Optional[str] = Field(
        default="gpt-5.4",
        description="The name of the AI model that the agent will utilize for processing tasks and generating outputs. For example: gpt-4o, gpt-4o-mini, openai/o3-mini",
    )
    auto_generate_prompt: Optional[bool] = Field(
        default=False,
        description="A flag indicating whether the agent should automatically create prompts based on the task requirements.",
    )
    max_tokens: Optional[int] = Field(
        default=8192,
        description="The maximum number of tokens that the agent is allowed to generate in its responses, limiting output length.",
    )
    temperature: Optional[float] = Field(
        default=0.5,
        description="A parameter that controls the randomness of the agent's output; lower values result in more deterministic responses.",
    )
    role: Optional[str] = Field(
        default="worker",
        description="The designated role of the agent within the swarm, which influences its behavior and interaction with other agents.",
    )
    max_loops: Optional[int] = Field(
        default=1,
        description="The maximum number of times the agent is allowed to repeat its task, enabling iterative processing if necessary.",
    )
    tools_list_dictionary: Optional[List[Dict[Any, Any]]] = Field(
        default=None,
        description="A dictionary of tools that the agent can use to complete its task.",
    )
    mcp_url: Optional[str] = Field(
        default=None,
        description="The URL of the MCP server that the agent can use to complete its task.",
    )

    class Config:
        arbitrary_types_allowed = True
