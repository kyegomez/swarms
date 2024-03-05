# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

# from swarms import ToolAgent
from swarms.utils.json_utils import base_model_schema_to_json

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-12b",
    load_in_4bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")


class Schema(BaseModel):
    name: str
    agent: int
    is_student: bool
    courses: list[str]


json_schema = str(base_model_schema_to_json(Schema))
print(json_schema)

# # Define the task to generate a person's information
# task = (
#     "Generate a person's information based on the following schema:"
# )

# # Create an instance of the ToolAgent class
# agent = ToolAgent(
#     name="dolly-function-agent",
#     description="Ana gent to create a child data",
#     model=model,
#     tokenizer=tokenizer,
#     json_schema=json_schema,
# )

# # Run the agent to generate the person's information
# generated_data = agent.run(task)

# # Print the generated data
# print(f"Generated data: {generated_data}")
