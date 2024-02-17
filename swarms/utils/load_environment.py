from dotenv import load_dotenv
import os

# Load the environment variables
def load_environment():
    load_dotenv()
    # Get the API key from the environment
    api_key = os.environ.get("OPENAI_API_KEY")

    return api_key, os.environ

