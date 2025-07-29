"""
Simple GraphWorkflow Examples

Quick examples demonstrating basic GraphWorkflow functionality.
These examples are designed to be easy to run and understand.
"""

import asyncio
import os
import sys

# Add the parent directory to the path so we can import from swarms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow, Node, Edge, NodeType, EdgeType

# Check for API key in environment variables
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
    print("   Please set your API key: export OPENAI_API_KEY='your-api-key-here'")
    print("   Or set it in your environment variables.")


async def example_1_basic_workflow():
    """Example 1: Basic workflow with two simple tasks."""
    print("\n🔧 Example 1: Basic Workflow")
    print("-" * 40)
    
    # Create workflow
    workflow = GraphWorkflow(name="Basic Example")
    
    # Define simple functions
    def task_1(**kwargs):
        return {"message": "Hello from Task 1", "data": [1, 2, 3]}
    
    def task_2(**kwargs):
        message = kwargs.get('message', '')
        data = kwargs.get('data', [])
        return {"final_result": f"{message} - Processed {len(data)} items"}
    
    # Create nodes
    node1 = Node(
        id="task_1",
        type=NodeType.TASK,
        callable=task_1,
        output_keys=["message", "data"]
    )
    
    node2 = Node(
        id="task_2",
        type=NodeType.TASK,
        callable=task_2,
        required_inputs=["message", "data"],
        output_keys=["final_result"]
    )
    
    # Add nodes and edges
    workflow.add_node(node1)
    workflow.add_node(node2)
    workflow.add_edge(Edge(source="task_1", target="task_2"))
    
    # Set entry and end points
    workflow.set_entry_points(["task_1"])
    workflow.set_end_points(["task_2"])
    
    # Run workflow
    result = await workflow.run("Basic workflow example")
    print(f"Result: {result['context_data']['final_result']}")
    
    return result


async def example_2_agent_workflow():
    """Example 2: Workflow with AI agents."""
    print("\n🤖 Example 2: Agent Workflow")
    print("-" * 40)
    
    # Create agents with cheapest models
    writer = Agent(
        agent_name="Writer",
        system_prompt="You are a creative writer. Write engaging content.",
        model_name="gpt-3.5-turbo"  # Cheaper model
    )
    
    editor = Agent(
        agent_name="Editor",
        system_prompt="You are an editor. Review and improve the content.",
        model_name="gpt-3.5-turbo"  # Cheaper model
    )
    
    # Create workflow
    workflow = GraphWorkflow(name="Content Creation")
    
    # Create nodes
    writer_node = Node(
        id="writer",
        type=NodeType.AGENT,
        agent=writer,
        output_keys=["content"],
        timeout=60.0
    )
    
    editor_node = Node(
        id="editor",
        type=NodeType.AGENT,
        agent=editor,
        required_inputs=["content"],
        output_keys=["edited_content"],
        timeout=60.0
    )
    
    # Add nodes and edges
    workflow.add_node(writer_node)
    workflow.add_node(editor_node)
    workflow.add_edge(Edge(source="writer", target="editor"))
    
    # Set entry and end points
    workflow.set_entry_points(["writer"])
    workflow.set_end_points(["editor"])
    
    # Run workflow
    result = await workflow.run("Write a short story about a robot learning to paint")
    print(f"Content created: {result['context_data']['edited_content'][:100]}...")
    
    return result


async def example_3_conditional_workflow():
    """Example 3: Workflow with conditional logic."""
    print("\n🔀 Example 3: Conditional Workflow")
    print("-" * 40)
    
    # Create workflow
    workflow = GraphWorkflow(name="Conditional Example")
    
    # Define functions
    def generate_number(**kwargs):
        import random
        number = random.randint(1, 100)
        return {"number": number}
    
    def check_even(**kwargs):
        number = kwargs.get('number', 0)
        return number % 2 == 0
    
    def process_even(**kwargs):
        number = kwargs.get('number', 0)
        return {"result": f"Even number {number} processed"}
    
    def process_odd(**kwargs):
        number = kwargs.get('number', 0)
        return {"result": f"Odd number {number} processed"}
    
    # Create nodes - using TASK type for condition since CONDITION doesn't exist
    nodes = [
        Node(id="generate", type=NodeType.TASK, callable=generate_number, output_keys=["number"]),
        Node(id="check", type=NodeType.TASK, callable=check_even, required_inputs=["number"], output_keys=["is_even"]),
        Node(id="even_process", type=NodeType.TASK, callable=process_even, required_inputs=["number"], output_keys=["result"]),
        Node(id="odd_process", type=NodeType.TASK, callable=process_odd, required_inputs=["number"], output_keys=["result"]),
    ]
    
    # Add nodes
    for node in nodes:
        workflow.add_node(node)
    
    # Add edges - simplified without conditional edges
    workflow.add_edge(Edge(source="generate", target="check"))
    workflow.add_edge(Edge(source="check", target="even_process"))
    workflow.add_edge(Edge(source="check", target="odd_process"))
    
    # Set entry and end points
    workflow.set_entry_points(["generate"])
    workflow.set_end_points(["even_process", "odd_process"])
    
    # Run workflow
    result = await workflow.run("Process a random number")
    print(f"Result: {result['context_data'].get('result', 'No result')}")
    
    return result


async def example_4_data_processing():
    """Example 4: Data processing workflow."""
    print("\n📊 Example 4: Data Processing")
    print("-" * 40)
    
    # Create workflow
    workflow = GraphWorkflow(name="Data Processing")
    
    # Define data processing functions
    def create_data(**kwargs):
        return {"raw_data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    
    def filter_data(**kwargs):
        data = kwargs.get('raw_data', [])
        filtered = [x for x in data if x % 2 == 0]
        return {"filtered_data": filtered}
    
    def calculate_stats(**kwargs):
        data = kwargs.get('filtered_data', [])
        return {
            "stats": {
                "count": len(data),
                "sum": sum(data),
                "average": sum(data) / len(data) if data else 0
            }
        }
    
    # Create nodes - using TASK type instead of DATA_PROCESSOR
    nodes = [
        Node(id="create", type=NodeType.TASK, callable=create_data, output_keys=["raw_data"]),
        Node(id="filter", type=NodeType.TASK, callable=filter_data, required_inputs=["raw_data"], output_keys=["filtered_data"]),
        Node(id="stats", type=NodeType.TASK, callable=calculate_stats, required_inputs=["filtered_data"], output_keys=["stats"]),
    ]
    
    # Add nodes
    for node in nodes:
        workflow.add_node(node)
    
    # Add edges
    workflow.add_edge(Edge(source="create", target="filter"))
    workflow.add_edge(Edge(source="filter", target="stats"))
    
    # Set entry and end points
    workflow.set_entry_points(["create"])
    workflow.set_end_points(["stats"])
    
    # Run workflow
    result = await workflow.run("Process and analyze data")
    print(f"Statistics: {result['context_data']['stats']}")
    
    return result


async def example_5_parallel_execution():
    """Example 5: Parallel execution workflow."""
    print("\n⚡ Example 5: Parallel Execution")
    print("-" * 40)
    
    # Create workflow
    workflow = GraphWorkflow(name="Parallel Example")
    
    # Define parallel tasks
    def task_a(**kwargs):
        import time
        time.sleep(0.1)  # Simulate work
        return {"result_a": "Task A completed"}
    
    def task_b(**kwargs):
        import time
        time.sleep(0.1)  # Simulate work
        return {"result_b": "Task B completed"}
    
    def task_c(**kwargs):
        import time
        time.sleep(0.1)  # Simulate work
        return {"result_c": "Task C completed"}
    
    def merge_results(**kwargs):
        results = []
        for key in ['result_a', 'result_b', 'result_c']:
            if key in kwargs:
                results.append(kwargs[key])
        return {"merged": results}
    
    # Create nodes - using TASK type instead of MERGE
    nodes = [
        Node(id="task_a", type=NodeType.TASK, callable=task_a, output_keys=["result_a"], parallel=True),
        Node(id="task_b", type=NodeType.TASK, callable=task_b, output_keys=["result_b"], parallel=True),
        Node(id="task_c", type=NodeType.TASK, callable=task_c, output_keys=["result_c"], parallel=True),
        Node(id="merge", type=NodeType.TASK, callable=merge_results, required_inputs=["result_a", "result_b", "result_c"], output_keys=["merged"]),
    ]
    
    # Add nodes
    for node in nodes:
        workflow.add_node(node)
    
    # Add edges (all parallel tasks feed into merge)
    workflow.add_edge(Edge(source="task_a", target="merge"))
    workflow.add_edge(Edge(source="task_b", target="merge"))
    workflow.add_edge(Edge(source="task_c", target="merge"))
    
    # Set entry and end points
    workflow.set_entry_points(["task_a", "task_b", "task_c"])
    workflow.set_end_points(["merge"])
    
    # Run workflow
    result = await workflow.run("Execute parallel tasks")
    print(f"Merged results: {result['context_data']['merged']}")
    
    return result


async def run_all_examples():
    """Run all simple examples."""
    print("🚀 Running GraphWorkflow Simple Examples")
    print("=" * 50)
    
    examples = [
        example_1_basic_workflow,
        example_2_agent_workflow,
        example_3_conditional_workflow,
        example_4_data_processing,
        example_5_parallel_execution,
    ]
    
    results = {}
    for i, example in enumerate(examples, 1):
        try:
            print(f"\n📝 Running Example {i}...")
            result = await example()
            results[f"example_{i}"] = result
            print(f"✅ Example {i} completed successfully")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            results[f"example_{i}"] = {"error": str(e)}
    
    print("\n" + "=" * 50)
    print("🎉 All examples completed!")
    print(f"✅ Successful: {sum(1 for r in results.values() if 'error' not in r)}")
    print(f"❌ Failed: {sum(1 for r in results.values() if 'error' in r)}")
    
    return results


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples()) 