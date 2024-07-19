"""
* WORKING

What this script does:
Structured output example

Requirements:
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY (this example works best with Openai bc it uses openai function calling structure)

Note:
If you are running playground examples in the project files directly (without swarms installed via PIP),
make sure to add the project root to your PYTHONPATH by running the following command in the project's root directory:
  'export PYTHONPATH=$(pwd):$PYTHONPATH'
"""

from pydantic import BaseModel, Field
from swarms import Agent, OpenAIChat


# Initialize the schema for the person's information
class Schema(BaseModel):
    """
    This is a pydantic class describing the format of a structured output
    """
    name: str = Field(..., title="Name of the person")
    agent: int = Field(..., title="Age of the person")
    is_student: bool = Field(..., title="Whether the person is a student")
    courses: list[str] = Field(
        ..., title="List of courses the person is taking"
    )

# Define the task to generate a person's information
task = "Generate a person's information based on the following schema:"

# Initialize the agent
agent = Agent(
    agent_name="Person Information Generator",
    system_prompt=(
        "Generate a person's information based on the following schema:"
    ),
    llm=OpenAIChat(),
    max_loops=1,
    streaming_on=True,
    verbose=True,
    # List of schemas that the agent can handle
    list_base_models=[Schema],
    agent_ops_on=True
)

# Run the agent to generate the person's information
generated_data = agent.run(task)

# Print the generated data
print(f"Generated data: {generated_data}")
