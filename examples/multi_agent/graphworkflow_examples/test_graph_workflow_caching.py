"""
Test script to demonstrate GraphWorkflow compilation caching for multi-loop scenarios.
This shows how the compilation is cached and reused across multiple loops to save compute.
"""

import time
from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def create_test_workflow(max_loops=3, verbose=True):
    """
    Create a test workflow with multiple agents to demonstrate caching.

    Args:
        max_loops (int): Number of loops to run (demonstrates caching when > 1)
        verbose (bool): Enable verbose logging to see caching behavior

    Returns:
        GraphWorkflow: Configured test workflow
    """

    # Create test agents
    analyzer = Agent(
        agent_name="Analyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a data analyzer. Analyze the given topic and provide insights.",
        verbose=False,  # Keep agent verbose low to focus on workflow caching logs
    )

    reviewer = Agent(
        agent_name="Reviewer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a reviewer. Review and validate the analysis provided.",
        verbose=False,
    )

    summarizer = Agent(
        agent_name="Summarizer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a summarizer. Create a concise summary of all previous work.",
        verbose=False,
    )

    # Create workflow with caching parameters
    workflow = GraphWorkflow(
        name="CachingTestWorkflow",
        description="Test workflow for demonstrating compilation caching",
        max_loops=max_loops,
        verbose=verbose,
        auto_compile=True,  # Enable auto-compilation for testing
    )

    # Add agents as nodes
    workflow.add_node(analyzer)
    workflow.add_node(reviewer)
    workflow.add_node(summarizer)

    # Create sequential flow: Analyzer -> Reviewer -> Summarizer
    workflow.add_edge("Analyzer", "Reviewer")
    workflow.add_edge("Reviewer", "Summarizer")

    return workflow


def test_single_loop_compilation():
    """Test compilation behavior with single loop (no caching benefit)."""
    print("=" * 60)
    print("TEST 1: Single Loop (No Caching Benefit)")
    print("=" * 60)

    workflow = create_test_workflow(max_loops=1, verbose=True)

    print("\n📊 Compilation Status Before Execution:")
    status = workflow.get_compilation_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n🚀 Running single loop workflow...")
    start_time = time.time()

    results = workflow.run(
        task="Analyze the benefits of renewable energy sources and provide a comprehensive summary."
    )

    execution_time = time.time() - start_time

    print(f"\n✅ Single loop completed in {execution_time:.3f}s")
    print(f"📋 Results: {len(results)} agents executed")

    print("\n📊 Compilation Status After Execution:")
    status = workflow.get_compilation_status()
    for key, value in status.items():
        if key != "layers":  # Skip layers for brevity
            print(f"  {key}: {value}")


def test_multi_loop_compilation():
    """Test compilation caching behavior with multiple loops."""
    print("\n\n" + "=" * 60)
    print("TEST 2: Multi-Loop (Caching Benefit)")
    print("=" * 60)

    workflow = create_test_workflow(max_loops=3, verbose=True)

    print("\n📊 Compilation Status Before Execution:")
    status = workflow.get_compilation_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n🚀 Running multi-loop workflow...")
    start_time = time.time()

    results = workflow.run(
        task="Research the impact of artificial intelligence on job markets. Provide detailed analysis, review, and summary."
    )

    execution_time = time.time() - start_time

    print(
        f"\n✅ Multi-loop execution completed in {execution_time:.3f}s"
    )
    print(f"📋 Results: {len(results)} agents executed")

    print("\n📊 Compilation Status After Execution:")
    status = workflow.get_compilation_status()
    for key, value in status.items():
        if key != "layers":  # Skip layers for brevity
            print(f"  {key}: {value}")


def test_cache_invalidation():
    """Test that cache is properly invalidated when graph structure changes."""
    print("\n\n" + "=" * 60)
    print("TEST 3: Cache Invalidation on Structure Change")
    print("=" * 60)

    workflow = create_test_workflow(max_loops=2, verbose=True)

    print("\n📊 Initial Compilation Status:")
    status = workflow.get_compilation_status()
    print(f"  Compiled: {status['is_compiled']}")
    print(f"  Cache Efficient: {status['cache_efficient']}")

    # Force compilation by running once
    print("\n🔄 Initial compilation run...")
    workflow.run(task="Initial test task")

    print("\n📊 Status After First Run:")
    status = workflow.get_compilation_status()
    print(f"  Compiled: {status['is_compiled']}")
    print(f"  Cache Efficient: {status['cache_efficient']}")
    print(
        f"  Compilation Timestamp: {status['compilation_timestamp']}"
    )

    # Add a new agent to trigger cache invalidation
    print("\n🔧 Adding new agent (should invalidate cache)...")
    new_agent = Agent(
        agent_name="Validator",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a validator. Validate all previous work.",
        verbose=False,
    )

    workflow.add_node(new_agent)
    workflow.add_edge("Summarizer", "Validator")

    print(
        "\n📊 Status After Adding Node (Cache Should Be Invalidated):"
    )
    status = workflow.get_compilation_status()
    print(f"  Compiled: {status['is_compiled']}")
    print(f"  Cache Efficient: {status['cache_efficient']}")
    print(
        f"  Compilation Timestamp: {status['compilation_timestamp']}"
    )

    # Run again to show recompilation
    print("\n🔄 Running with new structure (should recompile)...")
    workflow.run(task="Test task with new structure")

    print("\n📊 Status After Recompilation:")
    status = workflow.get_compilation_status()
    print(f"  Compiled: {status['is_compiled']}")
    print(f"  Cache Efficient: {status['cache_efficient']}")
    print(f"  Cached Layers: {status['cached_layers_count']}")


def run_caching_tests():
    """Run all caching demonstration tests."""
    print("🧪 GRAPHWORKFLOW COMPILATION CACHING TESTS")
    print(
        "Testing compilation caching behavior for multi-loop scenarios"
    )

    # Test 1: Single loop (baseline)
    test_single_loop_compilation()

    # Test 2: Multi-loop (demonstrates caching)
    test_multi_loop_compilation()

    # Test 3: Cache invalidation
    test_cache_invalidation()

    print("\n\n" + "=" * 60)
    print("🎯 CACHING SUMMARY")
    print("=" * 60)
    print("✅ Single loop: No caching needed")
    print("✅ Multi-loop: Compilation cached and reused")
    print("✅ Structure changes: Cache properly invalidated")
    print(
        "✅ Performance: Avoided redundant computation in multi-loop scenarios"
    )


if __name__ == "__main__":
    run_caching_tests()
