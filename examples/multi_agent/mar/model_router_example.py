import os
from swarms.structs.model_router import ModelRouter
from dotenv import load_dotenv

load_dotenv()

model_router = ModelRouter(api_key=os.getenv("OPENAI_API_KEY"))

model_router.run(
    "What are the best ways to analyze macroeconomic data? Use openai gpt-4o models"
)
