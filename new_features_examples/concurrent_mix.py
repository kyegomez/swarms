import os

from swarm_models import OpenAIChat

from swarms import Agent, run_agents_with_tasks_concurrently

# Fetch the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize agents for different roles
delaware_ccorp_agent = Agent(
    agent_name="Delaware-CCorp-Hiring-Agent",
    system_prompt="""
    Create a comprehensive hiring description for a Delaware C Corporation, 
    including all relevant laws and regulations, such as the Delaware General 
    Corporation Law (DGCL) and the Delaware Corporate Law. Ensure the description 
    covers the requirements for hiring employees, contractors, and officers, 
    including the necessary paperwork, tax obligations, and benefits. Also, 
    outline the procedures for compliance with Delaware's employment laws, 
    including anti-discrimination laws, workers' compensation, and unemployment 
    insurance. Provide guidance on how to navigate the complexities of Delaware's 
    corporate law and ensure that all hiring practices are in compliance with 
    state and federal regulations.
    """,
    llm=model,
    max_loops=1,
    autosave=False,
    dashboard=False,
    verbose=True,
    output_type="str",
    artifacts_on=True,
    artifacts_output_path="delaware_ccorp_hiring_description.md",
    artifacts_file_extension=".md",
)

indian_foreign_agent = Agent(
    agent_name="Indian-Foreign-Hiring-Agent",
    system_prompt="""
    Create a comprehensive hiring description for an Indian or foreign country, 
    including all relevant laws and regulations, such as the Indian Contract Act, 
    the Indian Labour Laws, and the Foreign Exchange Management Act (FEMA). 
    Ensure the description covers the requirements for hiring employees, 
    contractors, and officers, including the necessary paperwork, tax obligations, 
    and benefits. Also, outline the procedures for compliance with Indian and 
    foreign employment laws, including anti-discrimination laws, workers' 
    compensation, and unemployment insurance. Provide guidance on how to navigate 
    the complexities of Indian and foreign corporate law and ensure that all hiring 
    practices are in compliance with state and federal regulations. Consider the 
    implications of hiring foreign nationals and the requirements for obtaining 
    necessary visas and work permits.
    """,
    llm=model,
    max_loops=1,
    autosave=False,
    dashboard=False,
    verbose=True,
    output_type="str",
    artifacts_on=True,
    artifacts_output_path="indian_foreign_hiring_description.md",
    artifacts_file_extension=".md",
)

# List of agents and corresponding tasks
agents = [delaware_ccorp_agent, indian_foreign_agent]
tasks = [
    """
    Create a comprehensive hiring description for an Agent Engineer, including 
    required skills and responsibilities. Ensure the description covers the 
    necessary technical expertise, such as proficiency in AI/ML frameworks, 
    programming languages, and data structures. Outline the key responsibilities, 
    including designing and developing AI agents, integrating with existing systems, 
    and ensuring scalability and performance.
    """,
    """
    Generate a detailed job description for a Prompt Engineer, including 
    required skills and responsibilities. Ensure the description covers the 
    necessary technical expertise, such as proficiency in natural language processing, 
    machine learning, and software development. Outline the key responsibilities, 
    including designing and optimizing prompts for AI systems, ensuring prompt 
    quality and consistency, and collaborating with cross-functional teams.
    """,
]

# Run agents with tasks concurrently
results = run_agents_with_tasks_concurrently(
    agents,
    tasks,
    all_cores=True,
    device="cpu",
)

# Print the results
for result in results:
    print(result)
