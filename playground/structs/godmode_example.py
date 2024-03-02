import os

from dotenv import load_dotenv

from swarms import ModelParallelizer
from swarms.models import Anthropic, Gemini, Mixtral, OpenAIChat

load_dotenv()

# API Keys
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the models
llm = OpenAIChat(openai_api_key=openai_api_key)
anthropic = Anthropic(anthropic_api_key=anthropic_api_key)
mixtral = Mixtral()
gemini = Gemini(gemini_api_key=gemini_api_key)

# Initialize the parallelizer
llms = [llm, anthropic, mixtral, gemini]
parallelizer = ModelParallelizer(llms)

# Set the task
task = "Generate a 10,000 word blog on health and wellness."

# Run the task
out = parallelizer.run(task)

# Print the responses 1 by 1
for i in range(len(out)):
    print(f"Response from LLM {i}: {out[i]}")
