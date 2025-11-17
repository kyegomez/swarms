from pydantic import BaseModel, Field
from typing import Optional


class AgentConfiguration(BaseModel):
    """
    Comprehensive configuration schema for autonomous agent creation and management.

    This Pydantic model defines all the necessary parameters to create, configure,
    and manage an autonomous agent with specific behaviors, capabilities, and constraints.
    It enables dynamic agent generation with customizable properties and allows
    arbitrary additional fields for extensibility.

    All fields are required with no defaults, forcing explicit configuration of the agent.
    The schema supports arbitrary additional parameters through the extra='allow' configuration.

    Attributes:
        agent_name: Unique identifier name for the agent
        agent_description: Detailed description of the agent's purpose and capabilities
        system_prompt: Core system prompt that defines the agent's behavior and personality
        max_loops: Maximum number of reasoning loops the agent can perform
        dynamic_temperature_enabled: Whether to enable dynamic temperature adjustment
        model_name: The specific LLM model to use for the agent
        safety_prompt_on: Whether to enable safety prompts and guardrails
        temperature: Controls response randomness and creativity
        max_tokens: Maximum tokens in a single response
        context_length: Maximum conversation context length
        frequency_penalty: Penalty for token frequency to reduce repetition
        presence_penalty: Penalty for token presence to encourage diverse topics
        top_p: Nucleus sampling parameter for token selection
        tools: List of tools/functions available to the agent
    """

    agent_name: Optional[str] = Field(
        description="Unique and descriptive name for the agent. Should be clear, concise, and indicative of the agent's purpose or domain expertise.",
    )

    agent_description: Optional[str] = Field(
        description="Comprehensive description of the agent's purpose, capabilities, expertise area, and intended use cases. This helps users understand what the agent can do and when to use it.",
    )

    system_prompt: Optional[str] = Field(
        description="The core system prompt that defines the agent's personality, behavior, expertise, and response style. This is the foundational instruction that shapes how the agent interacts and processes information.",
    )

    max_loops: Optional[int] = Field(
        description="Maximum number of reasoning loops or iterations the agent can perform when processing complex tasks. Higher values allow for more thorough analysis but consume more resources.",
    )

    dynamic_temperature_enabled: Optional[bool] = Field(
        description="Whether to enable dynamic temperature adjustment during conversations. When enabled, the agent can adjust its creativity/randomness based on the task context - lower for factual tasks, higher for creative tasks.",
    )

    model_name: Optional[str] = Field(
        description="The specific language model to use for this agent. Should be a valid model identifier that corresponds to available LLM models in the system.",
    )

    safety_prompt_on: Optional[bool] = Field(
        description="Whether to enable safety prompts and content guardrails. When enabled, the agent will have additional safety checks to prevent harmful, biased, or inappropriate responses.",
    )

    temperature: Optional[float] = Field(
        description="Controls the randomness and creativity of the agent's responses. Lower values (0.0-0.3) for more focused and deterministic responses, higher values (0.7-1.0) for more creative and varied outputs.",
    )

    max_tokens: Optional[int] = Field(
        description="Maximum number of tokens the agent can generate in a single response. Controls the length and detail of agent outputs.",
    )

    context_length: Optional[int] = Field(
        description="Maximum context length the agent can maintain in its conversation memory. Affects how much conversation history the agent can reference.",
    )

    task: Optional[str] = Field(
        description="The task that the agent will perform.",
    )

    class Config:
        """Pydantic model configuration."""

        extra = "allow"  # Allow arbitrary additional fields
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        arbitrary_types_allowed = True  # Allow arbitrary types
