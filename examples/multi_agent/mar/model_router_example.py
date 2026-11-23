from swarms.structs.model_router import ModelRouter
from dotenv import load_dotenv

load_dotenv()

model_router = ModelRouter()

model_router.run(
    "What are the best ways to analyze macroeconomic data? Use openai gpt-4o models"
)
