from swarms.structs.agent import Agent

# Initialize the agent
agent = Agent(
    agent_name="Clinical-Documentation-Agent",
    agent_description="Specialized agent for clinical documentation and "
                     "medical record analysis",
    system_prompt="You are a clinical documentation specialist with expertise "
                 "in medical terminology, SOAP notes, and healthcare "
                 "documentation standards. You help analyze and improve "
                 "clinical documentation for accuracy, completeness, and "
                 "compliance.",
    max_loops=1,
    model_name="claude-opus-4-20250514",
    dynamic_temperature_enabled=True,
    output_type="final",
)

print(agent.run("what are the best ways to diagnose the flu?"))
