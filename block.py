import os

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the models, structs, and telemetry modules
from swarms import (
    Gemini,
    GPT4VisionAPI,
    Mixtral,
    OpenAI,
    ToolAgent,
    BlocksList,
)

# Load the environment variables
load_dotenv()

# Get the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Tool Agent
model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-12b"
)
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {"type": "array", "items": {"type": "string"}},
    },
}
toolagent = ToolAgent(
    model=model, tokenizer=tokenizer, json_schema=json_schema
)

# Blocks List which enables you to build custom swarms by adding classes or functions
swarm = BlocksList(
    "SocialMediaSwarm",
    "A swarm of social media agents",
    [
        OpenAI(openai_api_key=openai_api_key),
        Mixtral(),
        GPT4VisionAPI(openai_api_key=openai_api_key),
        Gemini(gemini_api_key=gemini_api_key),
    ],
)


# Add the new block to the swarm
swarm.add(toolagent)

# Remove a block from the swarm
swarm.remove(toolagent)

# Update a block in the swarm
swarm.update(toolagent)

# Get a block at a specific index
block_at_index = swarm.get(0)

# Get all blocks in the swarm
all_blocks = swarm.get_all()

# Get blocks by name
openai_blocks = swarm.get_by_name("OpenAI")

# Get blocks by type
gpt4_blocks = swarm.get_by_type("GPT4VisionAPI")

# Get blocks by ID
block_by_id = swarm.get_by_id(toolagent.id)

# Get blocks by parent
blocks_by_parent = swarm.get_by_parent(swarm)

# Get blocks by parent ID
blocks_by_parent_id = swarm.get_by_parent_id(swarm.id)

# Get blocks by parent name
blocks_by_parent_name = swarm.get_by_parent_name(swarm.name)

# Get blocks by parent type
blocks_by_parent_type = swarm.get_by_parent_type(type(swarm).__name__)

# Get blocks by parent description
blocks_by_parent_description = swarm.get_by_parent_description(
    swarm.description
)

# Run the block in the swarm
inference = swarm.run_block(toolagent, "Hello World")
print(inference)