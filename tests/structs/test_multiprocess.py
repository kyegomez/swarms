import asyncio
import time
from swarms.structs.agent import Agent
from swarms.structs.multi_process_workflow import MultiProcessWorkflow


def create_test_agent(name: str) -> Agent:
    """Create a test agent that simply returns its input with a timestamp"""
    return Agent(
        agent_name=name,
        system_prompt=f"Test prompt for {name}",
        model_name="gpt-4o-mini",
        max_loops=1,
    )


def test_initialization():
    """Test basic workflow initialization"""
    print("\n=== Testing Workflow Initialization ===")
    try:
        agents = [create_test_agent(f"agent{i}") for i in range(3)]
        workflow = MultiProcessWorkflow(max_workers=2, agents=agents)

        print("✓ Created workflow with configuration:")
        print(f"  - Max workers: {workflow.max_workers}")
        print(f"  - Number of agents: {len(workflow.agents)}")
        print(f"  - Autosave: {workflow.autosave}")
        print("✓ Initialization test passed")
    except Exception as e:
        print(f"✗ Initialization test failed: {str(e)}")
        raise


def test_execute_task():
    """Test execution of a single task"""
    print("\n=== Testing Task Execution ===")
    try:
        agents = [create_test_agent("test_agent")]
        workflow = MultiProcessWorkflow(agents=agents)

        test_task = "Return this message with timestamp"
        result = workflow.execute_task(test_task)

        print("✓ Task executed successfully")
        print(f"  - Input task: {test_task}")
        print(f"  - Result: {result}")
        print("✓ Task execution test passed")
    except Exception as e:
        print(f"✗ Task execution test failed: {str(e)}")
        raise


def test_parallel_run():
    """Test parallel execution of tasks"""
    print("\n=== Testing Parallel Run ===")
    try:
        agents = [create_test_agent(f"agent{i}") for i in range(3)]
        workflow = MultiProcessWorkflow(max_workers=2, agents=agents)

        test_task = "Process this in parallel"
        results = workflow.run(test_task)

        print("✓ Parallel execution completed")
        # print(f"  - Number of results: {len(results)}")
        print(f"  - Results: {results}")
        print("✓ Parallel run test passed")
    except Exception as e:
        print(f"✗ Parallel run test failed: {str(e)}")
        raise


async def test_async_run():
    """Test asynchronous execution of tasks"""
    print("\n=== Testing Async Run ===")
    try:
        agents = [create_test_agent(f"agent{i}") for i in range(3)]
        workflow = MultiProcessWorkflow(max_workers=2, agents=agents)

        test_task = "Process this asynchronously"
        results = await workflow.async_run(test_task)

        print("✓ Async execution completed")
        print(f"  - Number of results: {len(results)}")
        print(f"  - Results: {results}")
        print("✓ Async run test passed")
    except Exception as e:
        print(f"✗ Async run test failed: {str(e)}")
        raise


def test_batched_run():
    """Test batch execution of tasks"""
    print("\n=== Testing Batched Run ===")
    try:
        agents = [create_test_agent(f"agent{i}") for i in range(2)]
        workflow = MultiProcessWorkflow(max_workers=2, agents=agents)

        tasks = [f"Batch task {i}" for i in range(5)]
        results = workflow.batched_run(tasks, batch_size=2)

        print("✓ Batch execution completed")
        print(f"  - Number of tasks: {len(tasks)}")
        print("  - Batch size: 2")
        print(f"  - Results: {results}")
        print("✓ Batched run test passed")
    except Exception as e:
        print(f"✗ Batched run test failed: {str(e)}")
        raise


def test_concurrent_run():
    """Test concurrent execution of tasks"""
    print("\n=== Testing Concurrent Run ===")
    try:
        agents = [create_test_agent(f"agent{i}") for i in range(2)]
        workflow = MultiProcessWorkflow(max_workers=2, agents=agents)

        tasks = [f"Concurrent task {i}" for i in range(4)]
        results = workflow.concurrent_run(tasks)

        print("✓ Concurrent execution completed")
        print(f"  - Number of tasks: {len(tasks)}")
        print(f"  - Results: {results}")
        print("✓ Concurrent run test passed")
    except Exception as e:
        print(f"✗ Concurrent run test failed: {str(e)}")
        raise


def test_error_handling():
    """Test error handling in workflow"""
    print("\n=== Testing Error Handling ===")
    try:
        # Create workflow with no agents to trigger error
        workflow = MultiProcessWorkflow(max_workers=2, agents=None)
        result = workflow.execute_task(
            "This should handle the error gracefully"
        )

        print("✓ Error handled gracefully")
        print(f"  - Result when no agents: {result}")
        print("✓ Error handling test passed")
    except Exception as e:
        print(f"✗ Error handling test failed: {str(e)}")
        raise


async def run_all_tests():
    """Run all tests"""
    print("\n=== Starting MultiProcessWorkflow Test Suite ===")
    start_time = time.time()

    try:
        # Run synchronous tests
        test_initialization()
        test_execute_task()
        test_parallel_run()
        test_batched_run()
        test_concurrent_run()
        test_error_handling()

        # Run async test
        await test_async_run()

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print("\n=== Test Suite Completed Successfully ===")
        print(f"Time taken: {duration} seconds")

    except Exception as e:
        print("\n=== Test Suite Failed ===")
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
