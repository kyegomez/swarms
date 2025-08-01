# demo_validation.py

from swarms.structs.agent import Agent
from swarms.structs.graph_workflow import GraphWorkflow
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Create simple workflow
print("Creating simple workflow...")
wf = GraphWorkflow(name="Demo-Workflow", verbose=True)


agent1 = Agent(agent_name="DataCollector", model_name="claude-3-7-sonnet-20250219")
agent2 = Agent(agent_name="Analyzer", model_name="claude-3-7-sonnet-20250219")
agent3 = Agent(agent_name="Reporter", model_name="claude-3-7-sonnet-20250219")
agent4 = Agent(agent_name="Isolated", model_name="claude-3-7-sonnet-20250219")  # Isolated node


wf.add_node(agent1)
wf.add_node(agent2)
wf.add_node(agent3)
wf.add_node(agent4)  # Add isolated node


# Add edges
wf.add_edge("DataCollector", "Analyzer")
wf.add_edge("Analyzer", "Reporter")


print("\nValidate workflow (without auto-fix):")
result = wf.validate()
print(f"Workflow is valid: {result['is_valid']}")
print(f"Warnings: {result['warnings']}")
print(f"Errors: {result['errors']}")


print("\nValidate workflow (with auto-fix enabled):")
result = wf.validate(auto_fix=True)
print(f"Workflow is valid: {result['is_valid']}")
print(f"Warnings: {result['warnings']}")
print(f"Errors: {result['errors']}")
print(f"Fixed: {result['fixed']}")


# Create workflow with cycles
print("\n\nCreating workflow with cycles...")
wf2 = GraphWorkflow(name="Cyclic-Workflow", verbose=True)


wf2.add_node(Agent(agent_name="A", model_name="claude-3-7-sonnet-20250219"))
wf2.add_node(Agent(agent_name="B", model_name="claude-3-7-sonnet-20250219"))
wf2.add_node(Agent(agent_name="C", model_name="claude-3-7-sonnet-20250219"))


wf2.add_edge("A", "B")
wf2.add_edge("B", "C")
wf2.add_edge("C", "A")  # Create cycle


print("\nValidate workflow with cycles:")
result = wf2.validate()
print(f"Workflow is valid: {result['is_valid']}")
print(f"Warnings: {result['warnings']}")
if "cycles" in result:
    print(f"Detected cycles: {result['cycles']}")