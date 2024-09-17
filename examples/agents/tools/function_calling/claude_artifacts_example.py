from swarm_models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field


# Pydantic is a data validation library that provides data validation and parsing using Python type hints.
class ClaudeArtifact(BaseModel):
    name: str = Field(
        ...,
        description="The name of the artifact",
    )
    plan: str = Field(
        ...,
        description="Plan for the artifact, Do I generate a new python file or do I modify an existing one?",
    )
    file_name_path: str = Field(
        ...,
        description="The path to the file to modify or create for example: 'game.py'",
    )
    content_of_file: str = Field(
        ...,
        description="The content of the file to modify or create ",
    )
    edit_count: int = Field(
        ...,
        description="The number of times to edit the file",
    )


# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're an artifact creator, you're purpose is to create an artifact with the user provided specifications. Think of relevant names, descriptions, and context windows for the artifact. You need to provide the name of the artifact, the system prompt for the artifact, the description of the artifact, the maximum number of tokens to generate in the API response, the temperature for the artifact, the context window for the artifact, and the model name for the artifact from huggingface.",
    max_tokens=3500,
    temperature=0.9,
    base_model=ClaudeArtifact,
    parallel_tool_calls=False,
)

out = model.run(
    "Create a game in python that has never been created before. Create a new form of gaming experience that has never been contemplated before."
)
print(out)
