from swarms import Agent, ConcurrentWorkflow

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
    model_name="gpt-4.1",
    max_loops=1,
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
    model_name="gpt-4.1",
    max_loops=1,
)

# List of agents and corresponding tasks
agents = [delaware_ccorp_agent, indian_foreign_agent]
task = """
    Create a comprehensive hiring description for an Agent Engineer, including 
    required skills and responsibilities. Ensure the description covers the 
    necessary technical expertise, such as proficiency in AI/ML frameworks, 
    programming languages, and data structures. Outline the key responsibilities, 
    including designing and developing AI agents, integrating with existing systems, 
    and ensuring scalability and performance.
    """

# Run agents with tasks concurrently
swarm = ConcurrentWorkflow(
    agents=agents,
    output_type="list",
)

print(
    swarm.run(
        task="what is the best state to incorporate a company in the USA?"
    )
)
