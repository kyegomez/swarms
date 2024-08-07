from swarms.models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    agent_name: str = Field(
        ...,
        description="The name of the agent",
    )
    system_prompt: str = Field(
        ...,
        description="The system prompt for the agent",
    )
    agent_description: str = Field(
        ...,
        description="The description of the agent",
    )
    max_tokens: int = Field(
        ...,
        description="The maximum number of tokens to generate in the API response",
    )
    temperature: float = Field(
        ...,
        description="A parameter that controls the randomness of the generated text",
    )
    context_window: int = Field(
        ...,
        description="The context window for the agent",
    )
    model_name: str = Field(
        ...,
        description="The model name for the agent from huggingface",
    )


# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're an agent creator, you're purpose is to create an agent with the user provided specifications",
    max_tokens=500,
    temperature=0.5,
    base_model=AgentSpec,
    parallel_tool_calls=False,
)


# The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
out = model.run("Create an agent for sentiment analysis")
print(out)
