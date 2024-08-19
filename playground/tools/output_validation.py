"""
* WORKING
What this script does:
Structured output example
Requirements:
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY (this example works best with Openai bc it uses openai function calling structure)
   
"""

################ Adding project root to PYTHONPATH ################################
# If you are running playground examples in the project files directly, use this: 

import sys
import os

sys.path.insert(0, os.getcwd())

################ Adding project root to PYTHONPATH ################################

from pydantic import BaseModel, Field
from swarms import Agent, OpenAIChat


# Initialize the schema for the person's information
class PersonInfo(BaseModel):
    """
    To create a PersonInfo, you need to return a JSON object with the following format:
    {
        "function_call": "PersonInfo",
        "parameters": {
            ...
        }
    }   
    """
    name: str = Field(..., title="Name of the person")
    age: int = Field(..., title="Age of the person")
    is_student: bool = Field(..., title="Whether the person is a student")
    courses: list[str] = Field(
        ..., title="List of courses the person is taking"
    )

# Initialize the agent
agent = Agent(
    agent_name="Person Information Generator",
    system_prompt=(
        "Generate a person's information"
    ),
    llm=OpenAIChat(),
    max_loops=1,
    verbose=True,
    # List of pydantic models that the agent can use
    list_base_models=[PersonInfo],
    output_validation=True
)

# Define the task to generate a person's information
task = "Generate a person's information for Paul Graham 56 years old and is a student at Harvard University and is taking 3 courses: Math, Science, and History."

# Run the agent to generate the person's information
generated_data = agent.run(task)

# Print the generated data
print(type(generated_data))
print(f"Generated data: {generated_data}")