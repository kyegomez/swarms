import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms.models import OpenAIChat

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    model_name="gpt-4",
    openai_api_key=api_key,
    openai_org_id=org_id,
    max_tokens=1000,
)
