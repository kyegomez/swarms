"""
Example 2: Technical Architecture Debate with Batch Processing

This example demonstrates using batched_run to process multiple technical
architecture questions, comparing different approaches to system design.
"""

from swarms import Agent, DebateWithJudge

# Create specialized technical agents
pro_agent = Agent(
    agent_name="Microservices-Pro",
    system_prompt=(
        "You are a software architecture expert advocating for microservices architecture. "
        "You present arguments focusing on scalability, independent deployment, "
        "technology diversity, and team autonomy. You use real-world examples and "
        "case studies to support your position."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

con_agent = Agent(
    agent_name="Monolith-Pro",
    system_prompt=(
        "You are a software architecture expert advocating for monolithic architecture. "
        "You present counter-arguments focusing on simplicity, reduced complexity, "
        "easier debugging, and lower operational overhead. You identify weaknesses "
        "in microservices approaches and provide compelling alternatives."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

judge_agent = Agent(
    agent_name="Architecture-Judge",
    system_prompt=(
        "You are a senior software architect evaluating architecture debates. "
        "You analyze both arguments considering factors like team size, project scale, "
        "complexity, operational capabilities, and long-term maintainability. "
        "You provide balanced synthesis that considers context-specific trade-offs."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the debate system
architecture_debate = DebateWithJudge(
    pro_agent=pro_agent,
    con_agent=con_agent,
    judge_agent=judge_agent,
    max_loops=2,  # Fewer loops for more focused technical debates
    output_type="str-all-except-first",
    verbose=True,
)

# Define multiple architecture questions
architecture_questions = [
    "Should a startup with 5 developers use microservices or monolithic architecture?",
    "Is serverless architecture better than containerized deployments for event-driven systems?",
    "Should a financial application use SQL or NoSQL databases for transaction processing?",
    "Is event-driven architecture superior to request-response for real-time systems?",
]

# Execute batch processing
results = architecture_debate.batched_run(architecture_questions)

# Display results
for result in results:
    print(result)
