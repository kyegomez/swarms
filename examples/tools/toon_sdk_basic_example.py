"""
Basic TOON SDK Usage Example

This example demonstrates the fundamentals of using TOON SDK
for token-optimized serialization in Swarms.

Key Concepts:
    - Connection configuration
    - Encoding JSON to TOON format
    - Decoding TOON back to JSON
    - Token compression metrics

Expected Output:
    - Original JSON: ~150 tokens
    - TOON format: ~75 tokens (50% reduction)
"""

from swarms.schemas.toon_schemas import TOONConnection
from swarms.tools.toon_sdk_client import (
    TOONSDKClient,
    encode_with_toon_sync,
    decode_with_toon_sync,
)
from swarms.utils.toon_formatter import (
    TOONFormatter,
    toon_encode,
    toon_decode,
)


def example_1_local_formatter():
    """
    Example 1: Use local TOON formatter (no API required).

    This is useful for:
    - Rapid prototyping
    - Offline development
    - Testing without SDK credentials
    """
    print("=" * 60)
    print("Example 1: Local TOON Formatter")
    print("=" * 60)

    # Sample data
    data = {
        "user": "Alice Johnson",
        "email": "alice@example.com",
        "age": 30,
        "address": "123 Main St, NYC",
        "status": "active",
        "metadata": {
            "last_login": "2025-01-15T10:30:00Z",
            "account_type": "premium",
        },
    }

    # Initialize formatter
    formatter = TOONFormatter(
        compact_keys=True,
        omit_null=True,
        indent=0,
    )

    # Encode to TOON
    toon_str = formatter.encode(data)
    print(f"\nOriginal JSON ({len(str(data))} chars):")
    print(data)
    print(f"\nTOON Format ({len(toon_str)} chars):")
    print(toon_str)

    # Decode back to JSON
    decoded = formatter.decode(toon_str)
    print("\nDecoded JSON:")
    print(decoded)

    # Compression metrics
    compression = formatter.estimate_compression_ratio(data)
    print(f"\nCompression Ratio: {compression:.1%}")

    # Quick convenience functions
    print("\n" + "=" * 60)
    print("Using convenience functions:")
    print("=" * 60)

    quick_toon = toon_encode(data)
    quick_json = toon_decode(quick_toon)
    print(f"Quick encode: {quick_toon}")
    print(f"Quick decode: {quick_json}")


def example_2_sdk_client():
    """
    Example 2: Use TOON SDK client with API (requires API key).

    This provides:
    - Official TOON encoding algorithms
    - Schema-aware optimizations
    - Higher compression ratios
    - Production-grade reliability
    """
    print("\n" + "=" * 60)
    print("Example 2: TOON SDK Client")
    print("=" * 60)

    # Configure connection
    connection = TOONConnection(
        url="https://api.toon-format.com/v1",
        api_key="your_toon_api_key_here",  # Replace with actual key
        serialization_format="toon",
        enable_compression=True,
        timeout=30,
    )

    # Sample data with nested structure
    data = {
        "project": {
            "name": "AI Research Initiative",
            "description": "Large-scale machine learning research",
            "team_members": [
                {"name": "Alice", "role": "Lead Researcher", "active": True},
                {"name": "Bob", "role": "Data Scientist", "active": True},
                {"name": "Charlie", "role": "Engineer", "active": False},
            ],
            "budget": 1000000,
            "start_date": "2025-01-01",
            "status": "active",
        }
    }

    # Synchronous encoding
    try:
        toon_str = encode_with_toon_sync(
            data=data,
            connection=connection,
            verbose=True,
        )

        print("\nTOON Encoded:")
        print(toon_str)

        # Synchronous decoding
        decoded = decode_with_toon_sync(
            toon_data=toon_str,
            connection=connection,
            verbose=True,
        )

        print("\nDecoded JSON:")
        print(decoded)

    except Exception as e:
        print("\nNote: This example requires a valid TOON API key.")
        print(f"Error: {e}")


async def example_3_async_sdk():
    """
    Example 3: Async TOON SDK usage for high-performance applications.

    Benefits:
    - Non-blocking I/O
    - Batch processing
    - Concurrent requests
    - Production scalability
    """
    print("\n" + "=" * 60)
    print("Example 3: Async TOON SDK")
    print("=" * 60)

    connection = TOONConnection(
        url="https://api.toon-format.com/v1",
        api_key="your_toon_api_key_here",
        serialization_format="toon",
    )

    # Sample data batch
    data_batch = [
        {"id": 1, "name": "Product A", "price": 29.99, "stock": 100},
        {"id": 2, "name": "Product B", "price": 49.99, "stock": 50},
        {"id": 3, "name": "Product C", "price": 19.99, "stock": 200},
    ]

    try:
        async with TOONSDKClient(connection=connection) as client:
            # Batch encode
            print("\nBatch Encoding...")
            toon_list = await client.batch_encode(data_batch)

            for i, toon_str in enumerate(toon_list):
                print(f"Product {i+1} TOON: {toon_str}")

            # Batch decode
            print("\nBatch Decoding...")
            decoded_list = await client.batch_decode(toon_list)

            for i, decoded in enumerate(decoded_list):
                print(f"Product {i+1} JSON: {decoded}")

    except Exception as e:
        print("\nNote: This example requires a valid TOON API key.")
        print(f"Error: {e}")


def example_4_llm_prompt_optimization():
    """
    Example 4: Optimize data for LLM prompts.

    Use Case:
    - Reduce token count in prompts
    - Fit more context within limits
    - Lower API costs
    - Faster processing
    """
    print("\n" + "=" * 60)
    print("Example 4: LLM Prompt Optimization")
    print("=" * 60)

    # Simulate large dataset for LLM context
    user_data = [
        {
            "user_id": f"user_{i:04d}",
            "username": f"user{i}",
            "email": f"user{i}@example.com",
            "status": "active" if i % 2 == 0 else "inactive",
            "created_at": f"2025-01-{i%28+1:02d}T00:00:00Z",
            "last_login": f"2025-01-{i%28+1:02d}T12:00:00Z" if i % 2 == 0 else None,
        }
        for i in range(20)
    ]

    formatter = TOONFormatter()

    # Compare token counts
    import json
    json_str = json.dumps(user_data, separators=(",", ":"))
    toon_str = formatter.encode(user_data)

    print(f"\nStandard JSON: {len(json_str)} characters")
    print(f"TOON Format: {len(toon_str)} characters")
    print(f"Reduction: {(1 - len(toon_str)/len(json_str)):.1%}")

    # Show sample
    print("\nFirst 200 chars of JSON:")
    print(json_str[:200] + "...")
    print("\nFirst 200 chars of TOON:")
    print(toon_str[:200] + "...")


def example_5_schema_aware_compression():
    """
    Example 5: Schema-aware compression for structured data.

    Benefits:
    - Better compression for tabular data
    - Maintains type information
    - Optimized for repeated structures
    """
    print("\n" + "=" * 60)
    print("Example 5: Schema-Aware Compression")
    print("=" * 60)

    # Define schema
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "price": {"type": "number"},
            "in_stock": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "name", "price"],
    }

    # Sample products
    products = [
        {
            "id": 1,
            "name": "Laptop",
            "price": 999.99,
            "in_stock": True,
            "tags": ["electronics", "computers"],
        },
        {
            "id": 2,
            "name": "Mouse",
            "price": 29.99,
            "in_stock": True,
            "tags": ["electronics", "accessories"],
        },
        {
            "id": 3,
            "name": "Keyboard",
            "price": 79.99,
            "in_stock": False,
            "tags": ["electronics", "accessories"],
        },
    ]

    formatter = TOONFormatter(compact_keys=True, use_shorthand=True)

    print("\nWith Schema Awareness:")
    for product in products:
        toon = formatter.encode(product, schema=schema)
        print(f"Product {product['id']}: {toon}")

    # Estimate total compression
    import json
    json_size = len(json.dumps(products))
    toon_size = sum(len(formatter.encode(p, schema)) for p in products)

    print(f"\nTotal JSON: {json_size} chars")
    print(f"Total TOON: {toon_size} chars")
    print(f"Compression: {(1 - toon_size/json_size):.1%}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TOON SDK Examples")
    print("Token-Oriented Object Notation for Swarms")
    print("=" * 60)

    # Example 1: Local formatter (works offline)
    example_1_local_formatter()

    # Example 2: SDK client (requires API key)
    # Uncomment when you have a valid API key
    # example_2_sdk_client()

    # Example 3: Async SDK (requires API key)
    # Uncomment when you have a valid API key
    # asyncio.run(example_3_async_sdk())

    # Example 4: LLM prompt optimization
    example_4_llm_prompt_optimization()

    # Example 5: Schema-aware compression
    example_5_schema_aware_compression()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
