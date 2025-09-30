"""
Todo

- You send structured data to the swarm through the users form they make
- then connect rag for every agent using llama index to remember all the students data
- structured outputs
"""

import os
from dotenv import load_dotenv
from swarms.utils.litellm_wrapper import LiteLLM
from pydantic import BaseModel
from typing import List


class CollegeLog(BaseModel):
    college_name: str
    college_description: str
    college_admission_requirements: str


class CollegesRecommendation(BaseModel):
    colleges: List[CollegeLog]
    reasoning: str


load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
model = LiteLLM(
    model_name="groq/llama-3.1-70b-versatile",
    temperature=0.1,
)

function_caller = LiteLLM(
    model_name="gpt-4.1",
    system_prompt="""You are a college selection final decision maker. Your role is to:
    - Balance all relevant factors and stakeholder input.
    - Only return the output in the schema format.
    """,
    response_format=CollegesRecommendation,
    temperature=0.1,
)


print(
    function_caller.run(
        """
        Student Profile: Kye Gomez
        - GPA: 3.8
        - SAT: 1450
        - Interests: Computer Science, Robotics
        - Location Preference: East Coast
        - Extracurriculars: Robotics Club President, Math Team
        - Budget: Need financial aid
        - Preferred Environment: Medium-sized urban campus
    """
    )
)
