import os
from dotenv import load_dotenv
from swarms.models import Gemini
from swarms.structs import Agent

# Load environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Gemini API key not found. Please set it in your .env file")

# Initialize the first agent (Gemini Pro Vision) for image analysis
agent_analysis = Agent(
    llm=Gemini(
        temperature=0.8,
        model_name="gemini-pro-vision",
        gemini_api_key=api_key
    ),
    max_loops=1,
    autosave=True,
    dashboard=True,
)

# Initialize the second agent (Gemini Pro) for text-based tasks
agent_instruction = Agent(
    llm=Gemini(
        temperature=0.8,
        model_name="gemini-pro",
        gemini_api_key=api_key
    ),
    max_loops=1,
    autosave=True,
    dashboard=True,
)

# Task for the first agent: Image Analysis
task_analysis = """
Analyze this logo and summarize its content, then give a structured list of suggested improvements.
"""
img = 'swarmslogobanner.png'  # Replace with your image file name

# Run the first agent on the task
analysis_results = agent_analysis.run(task=task_analysis, img=img)

# Task for the second agent: Instructions for a better logo
task_instruction = f"""
Based on the analysis and suggestions for the logo:
{analysis_results}

Generate detailed instructions for creating an improved version of this logo.
"""

# Run the second agent on the task
instruction_results = agent_instruction.run(task=task_instruction)

............