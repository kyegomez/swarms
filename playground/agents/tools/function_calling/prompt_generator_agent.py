from swarms.models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field
from typing import Sequence


class PromptUseCase(BaseModel):
    use_case_name: str = Field(
        ...,
        description="The name of the use case",
    )
    use_case_description: str = Field(
        ...,
        description="The description of the use case",
    )


class PromptSpec(BaseModel):
    prompt_name: str = Field(
        ...,
        description="The name of the prompt",
    )
    prompt_description: str = Field(
        ...,
        description="The description of the prompt",
    )
    prompt: str = Field(
        ...,
        description="The prompt for the agent",
    )
    tags: str = Field(
        ...,
        description="The tags for the prompt such as sentiment, code, etc seperated by commas.",
    )
    use_cases: Sequence[PromptUseCase] = Field(
        ...,
        description="The use cases for the prompt",
    )


# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're an agent creator, you're purpose is to create system prompt for new LLM Agents for the user. Follow the best practices for creating a prompt such as making it direct and clear. Providing instructions and many-shot examples will help the agent understand the task better.",
    max_tokens=1000,
    temperature=0.5,
    base_model=PromptSpec,
    parallel_tool_calls=False,
)


# The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
out = model.run(
    "Create an prompt for generating quality rust code with instructions and examples."
)
print(out)
