from swarms.structs.agent import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter

# Example usage:
agents = [
    Agent(
        agent_name="ResearchAgent",
        agent_description="Specializes in researching topics and providing detailed, factual information",
        system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
        max_loops=1,
    ),
    Agent(
        agent_name="CodeExpertAgent",
        agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
        system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
        max_loops=1,
    ),
    Agent(
        agent_name="WritingAgent",
        agent_description="Skilled in creative and technical writing, content creation, and editing",
        system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
        max_loops=1,
    ),
]

models_to_test = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-5-nano-2025-08-07",
    "gpt-5-mini",
    "o4-mini", 
    "o3",            
    "claude-opus-4-20250514",   
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",   
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
]

task = "Use all the agents available to you to remake the Fibonacci function in Python, providing both an explanation and code."

model_logs = []

for model_name in models_to_test:
    print(f"\n--- Testing model: {model_name} ---")
    router_execute = MultiAgentRouter(
        agents=agents, temperature=0.5, model=model_name,
    )
    try:
        result = router_execute.run(task)
        print(f"Run completed successfully for {model_name}")
        model_logs.append({"model": model_name, "status": "✅ Success"})
    except Exception as e:
        print(f"An error occurred for {model_name}")
        model_logs.append({"model": model_name, "status": f"❌ Error: {e}"})

print("\n===== Model Run Summary =====")
for log in model_logs:
    print(f"{log['model']}: {log['status']}")

