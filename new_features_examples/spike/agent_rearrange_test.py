"""
Todo

- You send structured data to the swarm through the users form they make
- then connect rag for every agent using llama index to remember all the students data
- structured outputs
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel
from swarm_models import OpenAIChat, OpenAIFunctionCaller

from swarms import Agent, AgentRearrange


class CollegeLog(BaseModel):
    college_name: str
    college_description: str
    college_admission_requirements: str


class CollegesRecommendation(BaseModel):
    colleges: list[CollegeLog]
    reasoning: str


load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)

FINAL_AGENT_PROMPT = """
You are a college selection final decision maker. Your role is to:
    1. Synthesize all previous analyses and discussions
    2. Weigh competing factors and trade-offs
    3. Create a final ranked list of recommended colleges
    4. Provide clear rationale for each recommendation
    5. Include specific action items for each selected school
    6. Outline next steps in the application process

    Focus on creating actionable, well-reasoned final recommendations that
    balance all relevant factors and stakeholder input.

"""

function_caller = OpenAIFunctionCaller(
    system_prompt=FINAL_AGENT_PROMPT,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_model=CollegesRecommendation,
    parallel_tool_calls=True,
)

# Student Profile Analyzer Agent
profile_analyzer_agent = Agent(
    agent_name="Student-Profile-Analyzer",
    system_prompt="""You are an expert student profile analyzer. Your role is to:
    1. Analyze academic performance, test scores, and extracurricular activities
    2. Identify student's strengths, weaknesses, and unique qualities
    3. Evaluate personal statements and essays
    4. Assess leadership experiences and community involvement
    5. Determine student's preferences for college environment, location, and programs
    6. Create a comprehensive student profile summary

    Always consider both quantitative metrics (GPA, test scores) and qualitative aspects
    (personal growth, challenges overcome, unique perspectives).""",
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="profile_analyzer_agent.json",
    user_name="student",
    context_length=200000,
    output_type="string",
)

# College Research Agent
college_research_agent = Agent(
    agent_name="College-Research-Specialist",
    system_prompt="""You are a college research specialist. Your role is to:
    1. Maintain updated knowledge of college admission requirements
    2. Research academic programs, campus culture, and student life
    3. Analyze admission statistics and trends
    4. Evaluate college-specific opportunities and resources
    5. Consider financial aid availability and scholarship opportunities
    6. Track historical admission data and acceptance rates

    Focus on providing accurate, comprehensive information about each institution
    while considering both academic and cultural fit factors.""",
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="college_research_agent.json",
    user_name="researcher",
    context_length=200000,
    output_type="string",
)

# College Match Agent
college_match_agent = Agent(
    agent_name="College-Match-Maker",
    system_prompt="""You are a college matching specialist. Your role is to:
    1. Compare student profiles with college requirements
    2. Evaluate fit based on academic, social, and cultural factors
    3. Consider geographic preferences and constraints
    4. Assess financial fit and aid opportunities
    5. Create tiered lists of reach, target, and safety schools
    6. Explain the reasoning behind each match

    Always provide a balanced list with realistic expectations while
    considering both student preferences and admission probability.""",
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="college_match_agent.json",
    user_name="matcher",
    context_length=200000,
    output_type="string",
)

# Debate Moderator Agent
debate_moderator_agent = Agent(
    agent_name="Debate-Moderator",
    system_prompt="""You are a college selection debate moderator. Your role is to:
    1. Facilitate discussions between different perspectives
    2. Ensure all relevant factors are considered
    3. Challenge assumptions and biases
    4. Synthesize different viewpoints
    5. Guide the group toward consensus
    6. Document key points of agreement and disagreement

    Maintain objectivity while ensuring all important factors are thoroughly discussed
    and evaluated.""",
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="debate_moderator_agent.json",
    user_name="moderator",
    context_length=200000,
    output_type="string",
)

# Critique Agent
critique_agent = Agent(
    agent_name="College-Selection-Critic",
    system_prompt="""You are a college selection critic. Your role is to:
    1. Evaluate the strength of college matches
    2. Identify potential overlooked factors
    3. Challenge assumptions in the selection process
    4. Assess risks and potential drawbacks
    5. Provide constructive feedback on selections
    6. Suggest alternative options when appropriate

    Focus on constructive criticism that helps improve the final college list
    while maintaining realistic expectations.""",
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="critique_agent.json",
    user_name="critic",
    context_length=200000,
    output_type="string",
)

# Final Decision Agent
final_decision_agent = Agent(
    agent_name="Final-Decision-Maker",
    system_prompt="""
    You are a college selection final decision maker. Your role is to:
    1. Synthesize all previous analyses and discussions
    2. Weigh competing factors and trade-offs
    3. Create a final ranked list of recommended colleges
    4. Provide clear rationale for each recommendation
    5. Include specific action items for each selected school
    6. Outline next steps in the application process

    Focus on creating actionable, well-reasoned final recommendations that
    balance all relevant factors and stakeholder input.
    """,
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="final_decision_agent.json",
    user_name="decision_maker",
    context_length=200000,
    output_type="string",
)

# Initialize the Sequential Workflow
college_selection_workflow = AgentRearrange(
    name="college-selection-swarm",
    description="Comprehensive college selection and analysis system",
    max_loops=1,
    agents=[
        profile_analyzer_agent,
        college_research_agent,
        college_match_agent,
        debate_moderator_agent,
        critique_agent,
        final_decision_agent,
    ],
    output_type="all",
    flow=f"{profile_analyzer_agent.name} -> {college_research_agent.name} -> {college_match_agent.name} -> {debate_moderator_agent.name} -> {critique_agent.name} -> {final_decision_agent.name}",
)

# Example usage
if __name__ == "__main__":
    # Example student profile input
    student_profile = """
    Student Profile:
    - GPA: 3.8
    - SAT: 1450
    - Interests: Computer Science, Robotics
    - Location Preference: East Coast
    - Extracurriculars: Robotics Club President, Math Team
    - Budget: Need financial aid
    - Preferred Environment: Medium-sized urban campus
    """

    # Run the comprehensive college selection analysis
    result = college_selection_workflow.run(
        student_profile,
        no_use_clusterops=True,
    )
    print(result)
