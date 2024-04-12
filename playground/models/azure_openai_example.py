import os

from dotenv import load_dotenv

from swarms import AzureOpenAI

# Load the environment variables
load_dotenv()

# Create an instance of the AzureOpenAI class
model = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_ad_token=os.getenv("AZURE_OPENAI_AD_TOKEN"),
)

# Define the prompt
prompt = (
    "Analyze this load document and assess it for any risks and"
    " create a table in markdwon format."
)

# Generate a response
response = model(prompt)
print(response)
