from swarms.models.azure_openai_llm import AzureOpenAI

# Initialize Azure OpenAI
model = AzureOpenAI()

# Run the model
model(
    "Create a youtube script for a video on how to use the swarms"
    " framework"
)
