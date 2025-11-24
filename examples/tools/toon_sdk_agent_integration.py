"""
TOON SDK + Swarms Agent Integration Example

This example demonstrates advanced integration of TOON SDK with
Swarms Agent for token-optimized multi-agent workflows.

Key Features:
    - Agent with TOON-optimized prompts
    - Automatic token reduction for tool outputs
    - Multi-agent coordination with compressed messages
    - Production-ready error handling

Expected Benefits:
    - 30-60% reduction in prompt tokens
    - Lower API costs
    - Faster response times
    - More context within token limits
"""

import asyncio
from swarms import Agent
from swarms.schemas.toon_schemas import TOONConnection, TOONSerializationOptions
from swarms.tools.toon_sdk_client import TOONSDKClient
from swarms.utils.toon_formatter import TOONFormatter, optimize_for_llm


# Example 1: Agent with TOON-Optimized System Prompt
def example_1_toon_optimized_agent():
    """
    Create an agent with TOON-optimized system prompts and tool outputs.

    Benefits:
    - Reduced prompt tokens
    - More efficient context usage
    - Lower costs per request
    """
    print("=" * 60)
    print("Example 1: TOON-Optimized Agent")
    print("=" * 60)

    # Define a tool that returns large JSON data
    def get_user_database() -> dict:
        """
        Retrieve user database with 50 users.

        Returns:
            dict: User database with full profiles
        """
        return {
            "users": [
                {
                    "user_id": f"usr_{i:04d}",
                    "username": f"user{i}",
                    "email": f"user{i}@company.com",
                    "full_name": f"User {i}",
                    "department": ["Engineering", "Sales", "Marketing", "HR"][i % 4],
                    "status": "active" if i % 3 != 0 else "inactive",
                    "created_at": f"2024-{(i%12)+1:02d}-01",
                    "last_login": f"2025-01-{(i%28)+1:02d}",
                    "permissions": ["read", "write"] if i % 2 == 0 else ["read"],
                }
                for i in range(50)
            ],
            "total_count": 50,
            "active_count": 34,
            "departments": ["Engineering", "Sales", "Marketing", "HR"],
        }

    # Wrapper to apply TOON compression to tool output
    def get_user_database_toon() -> str:
        """Get user database with TOON compression."""
        data = get_user_database()
        formatter = TOONFormatter(compact_keys=True, omit_null=True)
        return formatter.encode(data)

    # Create agent with TOON-optimized tool
    agent = Agent(
        agent_name="Data-Analyst-Agent",
        model_name="gpt-4o",
        max_loops=1,
        tools=[get_user_database_toon],
        system_prompt="""You are a data analyst assistant.
When analyzing user data, provide insights on:
- Active vs inactive user ratios
- Department distribution
- Recent activity patterns

Use the get_user_database_toon tool which returns data in TOON format (compact notation).
Interpret the TOON format where 'usr' = user, 'eml' = email, 'sts' = status, etc.
""",
        streaming_on=False,
    )

    # Run analysis
    response = agent.run(
        "Analyze the user database and provide a summary of active users by department."
    )

    print("\nAgent Response:")
    print(response)

    # Show token savings
    import json
    regular_data = get_user_database()
    toon_data = get_user_database_toon()

    print(f"\n{'='*60}")
    print("Token Savings:")
    print(f"Regular JSON: ~{len(json.dumps(regular_data))} chars")
    print(f"TOON Format: ~{len(toon_data)} chars")
    print(f"Reduction: {(1 - len(toon_data)/len(json.dumps(regular_data))):.1%}")


# Example 2: Multi-Agent with TOON Message Passing
def example_2_multi_agent_toon():
    """
    Multi-agent system with TOON-compressed inter-agent messages.

    Architecture:
    - Data Collector Agent → TOON compression → Analyzer Agent
    - Reduced message overhead
    - Faster multi-agent coordination
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-Agent with TOON Messages")
    print("=" * 60)

    formatter = TOONFormatter()

    # Agent 1: Data Collector
    def collect_sales_data() -> dict:
        """Collect sales data from multiple regions."""
        return {
            "regions": {
                "North": {"revenue": 125000, "orders": 450, "growth": 15.5},
                "South": {"revenue": 98000, "orders": 380, "growth": 12.3},
                "East": {"revenue": 156000, "orders": 520, "growth": 18.2},
                "West": {"revenue": 142000, "orders": 490, "growth": 16.8},
            },
            "period": "Q1-2025",
            "currency": "USD",
        }

    collector_agent = Agent(
        agent_name="Data-Collector",
        model_name="gpt-4o",
        max_loops=1,
        tools=[collect_sales_data],
        system_prompt="""You are a data collection agent.
Collect sales data using the collect_sales_data tool.
Format your output as structured data only, no commentary.""",
    )

    # Agent 2: Data Analyzer (receives TOON-compressed data)
    analyzer_agent = Agent(
        agent_name="Data-Analyzer",
        model_name="gpt-4o",
        max_loops=1,
        system_prompt="""You are a sales analyst.
You receive data in TOON format (compressed notation).
Analyze the data and provide insights on:
- Top performing region
- Growth trends
- Revenue distribution""",
    )

    # Step 1: Collector gathers data
    print("\n[Step 1] Collector gathering data...")
    raw_data = collect_sales_data()
    print(f"Raw data collected: {len(str(raw_data))} chars")

    # Step 2: Compress with TOON
    print("\n[Step 2] Compressing with TOON...")
    toon_data = formatter.encode(raw_data)
    print(f"TOON compressed: {len(toon_data)} chars")
    print(f"Compression: {(1 - len(toon_data)/len(str(raw_data))):.1%}")

    # Step 3: Analyzer receives compressed data
    print("\n[Step 3] Analyzer processing TOON data...")
    analysis_prompt = f"""Analyze this sales data (TOON format):

{toon_data}

Provide insights on regional performance and growth trends."""

    analysis = analyzer_agent.run(analysis_prompt)

    print("\nAnalysis Result:")
    print(analysis)


# Example 3: TOON-Enabled Tool Registry
async def example_3_toon_tool_registry():
    """
    Register and use TOON-enabled tools from SDK.

    Benefits:
    - Automatic tool discovery
    - Schema-aware compression
    - OpenAI-compatible conversion
    """
    print("\n" + "=" * 60)
    print("Example 3: TOON Tool Registry")
    print("=" * 60)

    # Configure TOON connection
    connection = TOONConnection(
        url="https://api.toon-format.com/v1",
        api_key="your_api_key_here",
        enable_compression=True,
    )

    try:
        async with TOONSDKClient(connection=connection) as client:
            # List available TOON tools
            print("\nFetching TOON tools...")
            tools = await client.list_tools()

            print(f"\nFound {len(tools)} TOON tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    Compression: {tool.compression_ratio:.1%}")

            # Convert to OpenAI format for Agent
            openai_tools = client.get_tools_as_openai_format()

            # Create agent with TOON tools
            agent = Agent(
                agent_name="TOON-Enabled-Agent",
                model_name="gpt-4o",
                max_loops=1,
                tools=openai_tools,  # Use TOON-optimized tools
                system_prompt="""You have access to TOON-optimized tools.
These tools automatically compress data for efficient processing.
Use them to retrieve and analyze information.""",
            )

            print("\nAgent created with TOON tools!")

    except Exception as e:
        print(f"\nNote: Requires valid TOON API key")
        print(f"Error: {e}")


# Example 4: Production RAG with TOON
def example_4_rag_with_toon():
    """
    Retrieval-Augmented Generation with TOON compression.

    Use Case:
    - Compress retrieved documents
    - Fit more context in prompts
    - Reduce embedding storage
    """
    print("\n" + "=" * 60)
    print("Example 4: RAG with TOON Compression")
    print("=" * 60)

    # Simulate document retrieval
    documents = [
        {
            "doc_id": f"doc_{i:04d}",
            "title": f"Research Paper {i}",
            "content": f"This is the abstract of research paper {i}. " * 10,
            "authors": [f"Author {j}" for j in range(3)],
            "published": f"2024-{(i%12)+1:02d}-01",
            "citations": i * 10,
            "keywords": ["AI", "ML", "Research"],
        }
        for i in range(10)
    ]

    formatter = TOONFormatter()

    # Regular approach: Full JSON
    import json
    regular_context = json.dumps(documents, indent=2)

    # TOON approach: Compressed
    toon_context = formatter.encode(documents)

    print("\nContext Size Comparison:")
    print(f"Regular JSON: {len(regular_context)} chars (~{len(regular_context)//4} tokens)")
    print(f"TOON Format: {len(toon_context)} chars (~{len(toon_context)//4} tokens)")
    print(f"Tokens Saved: ~{(len(regular_context) - len(toon_context))//4} tokens")

    # Create RAG agent with TOON context
    rag_agent = Agent(
        agent_name="RAG-Agent",
        model_name="gpt-4o",
        max_loops=1,
        system_prompt=f"""You are a research assistant with access to compressed document context.

The following documents are provided in TOON format (compact notation):

{toon_context[:500]}...

Answer questions based on this context. Interpret TOON format where common abbreviations apply.""",
    )

    # Query
    response = rag_agent.run(
        "What are the most cited papers in this collection?"
    )

    print("\nRAG Response:")
    print(response)


# Example 5: Real-Time Optimization
def example_5_realtime_optimization():
    """
    Real-time TOON optimization for streaming responses.

    Use Case:
    - Optimize data on-the-fly
    - Streaming agent responses
    - Dynamic compression decisions
    """
    print("\n" + "=" * 60)
    print("Example 5: Real-Time TOON Optimization")
    print("=" * 60)

    formatter = TOONFormatter()

    def optimize_response(data: dict) -> str:
        """
        Optimize response data in real-time.

        Decides between TOON, JSON, or compact based on data characteristics.
        """
        # Calculate compression potential
        import json
        json_len = len(json.dumps(data))
        toon_len = len(formatter.encode(data))

        compression = (json_len - toon_len) / json_len

        # Decision logic
        if compression > 0.3:  # >30% savings
            format_used = "TOON"
            result = formatter.encode(data)
        elif json_len < 200:  # Small data
            format_used = "JSON"
            result = json.dumps(data, indent=2)
        else:
            format_used = "Compact JSON"
            result = json.dumps(data, separators=(",", ":"))

        print(f"\nOptimization Decision: {format_used}")
        print(f"Original: {json_len} chars")
        print(f"Optimized: {len(result)} chars")
        print(f"Savings: {compression:.1%}")

        return result

    # Test with different data sizes
    small_data = {"user": "Alice", "age": 30}
    large_data = {
        "users": [
            {"id": i, "name": f"User{i}", "email": f"u{i}@ex.com", "active": True}
            for i in range(20)
        ]
    }

    print("\nSmall Data Optimization:")
    optimize_response(small_data)

    print("\nLarge Data Optimization:")
    optimize_response(large_data)


def main():
    """Run all integration examples."""
    print("\n" + "=" * 60)
    print("TOON SDK + Swarms Agent Integration")
    print("Advanced Examples for Production Use")
    print("=" * 60)

    # Example 1: TOON-Optimized Agent
    try:
        example_1_toon_optimized_agent()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")

    # Example 2: Multi-Agent with TOON
    try:
        example_2_multi_agent_toon()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")

    # Example 3: TOON Tool Registry (requires async)
    # Uncomment when you have a valid API key
    # asyncio.run(example_3_toon_tool_registry())

    # Example 4: RAG with TOON
    try:
        example_4_rag_with_toon()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")

    # Example 5: Real-Time Optimization
    try:
        example_5_realtime_optimization()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")

    print("\n" + "=" * 60)
    print("Integration Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
