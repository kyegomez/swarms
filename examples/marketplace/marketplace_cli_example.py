"""
Swarms Marketplace CLI Examples

This file demonstrates how to use the Swarms marketplace CLI commands
to search, browse, and install agents from the Swarms marketplace.

Prerequisites:
1. Set your SWARMS_API_KEY environment variable:
   export SWARMS_API_KEY="your-api-key-here"

   Get your API key at: https://swarms.world/platform/api-keys

2. Install swarms:
   pip install swarms

Usage Examples (run these in your terminal):
"""

# =============================================================================
# CLI COMMAND EXAMPLES
# =============================================================================

CLI_EXAMPLES = """
# ─────────────────────────────────────────────────────────────────────────────
# 1. SEARCH FOR AGENTS
# ─────────────────────────────────────────────────────────────────────────────

# Search by keyword
swarms marketplace search --query "trading"

# Search by category
swarms marketplace search --category "finance"

# Search with multiple filters
swarms marketplace search --query "analysis" --category "data-analysis" --free-only

# Limit results
swarms marketplace search --query "automation" --limit 10


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIST ALL AGENTS
# ─────────────────────────────────────────────────────────────────────────────

# List all available agents
swarms marketplace list

# List only free agents
swarms marketplace list --free-only

# List agents in a specific category
swarms marketplace list --category "coding"

# List with custom limit
swarms marketplace list --limit 50


# ─────────────────────────────────────────────────────────────────────────────
# 3. VIEW AGENT DETAILS
# ─────────────────────────────────────────────────────────────────────────────

# Get detailed information about an agent
swarms marketplace info <agent-id>

# Example with a real ID:
# swarms marketplace info 550e8400-e29b-41d4-a716-446655440000


# ─────────────────────────────────────────────────────────────────────────────
# 4. INSTALL AGENTS
# ─────────────────────────────────────────────────────────────────────────────

# Install to current directory
swarms marketplace install <agent-id>

# Install to a specific directory
swarms marketplace install <agent-id> --output-dir ./my_agents

# Install to agents folder
swarms marketplace install <agent-id> --output-dir ./agents


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET HELP
# ─────────────────────────────────────────────────────────────────────────────

# Show marketplace help
swarms marketplace

# Show all available CLI commands
swarms help
"""

# =============================================================================
# PROGRAMMATIC USAGE (Python API)
# =============================================================================

def example_programmatic_usage():
    """
    Example of using the marketplace utilities programmatically in Python.
    """
    from swarms.utils.agent_marketplace import (
        query_agents,
        get_agent_by_id,
        install_agent,
        list_available_categories,
    )

    # List available categories
    categories = list_available_categories()
    print(f"Available categories: {categories}")

    # Search for agents
    results = query_agents(
        search="trading",
        category="finance",
        price_filter="free",
        limit=10,
    )
    print(f"Found agents: {results}")

    # Get agent details
    # agent = get_agent_by_id("your-agent-id-here")
    # print(f"Agent details: {agent}")

    # Install an agent
    # result = install_agent(
    #     agent_id="your-agent-id-here",
    #     output_dir="./my_agents"
    # )
    # print(f"Installation result: {result}")


# =============================================================================
# WORKFLOW EXAMPLE
# =============================================================================

WORKFLOW_EXAMPLE = """
TYPICAL WORKFLOW:

1. First, search for agents that match your needs:
   $ swarms marketplace search --query "customer service" --category "automation"

2. Review the list and find an agent you like. Note its ID.

3. Get more details about the agent:
   $ swarms marketplace info abc123-def456

4. Install the agent to your project:
   $ swarms marketplace install abc123-def456 --output-dir ./agents

5. The agent file will be created with all metadata and code.
   You can then import and use it in your project:

   from agents.customer_service_agent import agent
   result = agent.run("Help me with a customer inquiry")
"""

# =============================================================================
# AVAILABLE CATEGORIES
# =============================================================================

CATEGORIES = [
    "finance",
    "research",
    "coding",
    "content",
    "data-analysis",
    "automation",
    "customer-service",
    "healthcare",
    "legal",
    "marketing",
    "education",
    "general",
]


if __name__ == "__main__":
    print("=" * 70)
    print("SWARMS MARKETPLACE CLI EXAMPLES")
    print("=" * 70)
    print(CLI_EXAMPLES)
    print("\n" + "=" * 70)
    print("WORKFLOW EXAMPLE")
    print("=" * 70)
    print(WORKFLOW_EXAMPLE)
    print("\n" + "=" * 70)
    print("AVAILABLE CATEGORIES")
    print("=" * 70)
    for cat in CATEGORIES:
        print(f"  - {cat}")
    print("\n" + "=" * 70)
    print("For more info: https://docs.swarms.world")
    print("Get API key: https://swarms.world/platform/api-keys")
    print("=" * 70)
