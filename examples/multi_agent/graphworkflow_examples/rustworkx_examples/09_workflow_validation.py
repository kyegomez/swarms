from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

agent_a = Agent(
    agent_name="Agent-A",
    agent_description="Agent A",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

agent_b = Agent(
    agent_name="Agent-B",
    agent_description="Agent B",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

agent_c = Agent(
    agent_name="Agent-C",
    agent_description="Agent C",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

agent_isolated = Agent(
    agent_name="Agent-Isolated",
    agent_description="Isolated agent with no connections",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

workflow = GraphWorkflow(
    name="Validation-Workflow",
    description="Workflow for validation testing",
    backend="rustworkx",
    verbose=False,
)

workflow.add_node(agent_a)
workflow.add_node(agent_b)
workflow.add_node(agent_c)
workflow.add_node(agent_isolated)

workflow.add_edge(agent_a, agent_b)
workflow.add_edge(agent_b, agent_c)

validation_result = workflow.validate(auto_fix=False)
print(f"Valid: {validation_result['is_valid']}")
print(f"Warnings: {len(validation_result['warnings'])}")
print(f"Errors: {len(validation_result['errors'])}")

validation_result_fixed = workflow.validate(auto_fix=True)
print(
    f"After auto-fix - Valid: {validation_result_fixed['is_valid']}"
)
print(f"Fixed: {len(validation_result_fixed['fixed'])}")
print(f"Entry points: {workflow.entry_points}")
print(f"End points: {workflow.end_points}")

workflow_cycle = GraphWorkflow(
    name="Cycle-Test-Workflow",
    backend="rustworkx",
    verbose=False,
)

workflow_cycle.add_node(agent_a)
workflow_cycle.add_node(agent_b)
workflow_cycle.add_node(agent_c)

workflow_cycle.add_edge(agent_a, agent_b)
workflow_cycle.add_edge(agent_b, agent_c)
workflow_cycle.add_edge(agent_c, agent_a)

cycle_validation = workflow_cycle.validate(auto_fix=False)
print(f"Cycles detected: {len(cycle_validation.get('cycles', []))}")
