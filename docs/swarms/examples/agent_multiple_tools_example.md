# Agents with Multiple Tools

This tutorial demonstrates how to create powerful agents that leverage multiple tools working together. Learn how agents intelligently select, chain, and orchestrate multiple tools to accomplish complex tasks.

## Overview

When an agent has access to multiple tools, it can:

- **Select the right tool** for each subtask automatically
- **Chain tools together** to accomplish complex workflows
- **Combine different tool categories** (APIs, data processing, calculations)
- **Handle tool dependencies** where one tool's output feeds into another

## Prerequisites

- Python 3.8+
- Swarms library installed
- API keys for any external services you want to use

## Installation

```bash
pip install -U swarms requests
```

## Example 1: Financial Analysis Agent with Multiple Tools

This example shows an agent using multiple financial tools to perform comprehensive market analysis.

```python
import json
import requests
from swarms import Agent
from typing import List, Dict
import time

# Tool 1: Get cryptocurrency price
def get_coin_price(coin_id: str, vs_currency: str = "usd") -> str:
    """
    Get the current price of a specific cryptocurrency.

    Args:
        coin_id (str): The CoinGecko ID (e.g., 'bitcoin', 'ethereum')
        vs_currency (str): Target currency. Defaults to "usd".

    Returns:
        str: JSON formatted string with price and market data
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Tool 2: Get top cryptocurrencies
def get_top_cryptocurrencies(limit: int = 10, vs_currency: str = "usd") -> str:
    """
    Fetch the top cryptocurrencies by market capitalization.

    Args:
        limit (int): Number of coins to retrieve (1-250). Defaults to 10.
        vs_currency (str): Target currency. Defaults to "usd".

    Returns:
        str: JSON formatted string with top cryptocurrencies
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Tool 3: Calculate portfolio value
def calculate_portfolio_value(holdings: Dict[str, float]) -> str:
    """
    Calculate the total value of a cryptocurrency portfolio.

    Args:
        holdings (Dict[str, float]): Dictionary mapping coin IDs to quantities
                                    Example: {"bitcoin": 0.5, "ethereum": 2.0}

    Returns:
        str: JSON formatted string with portfolio breakdown and total value
    """
    try:
        portfolio_data = {}
        total_value = 0.0
        
        for coin_id, quantity in holdings.items():
            price_data = json.loads(get_coin_price(coin_id))
            if coin_id in price_data and "usd" in price_data[coin_id]:
                price = price_data[coin_id]["usd"]
                value = quantity * price
                portfolio_data[coin_id] = {
                    "quantity": quantity,
                    "price_usd": price,
                    "value_usd": value
                }
                total_value += value
                time.sleep(0.5)  # Rate limiting
        
        result = {
            "holdings": portfolio_data,
            "total_value_usd": round(total_value, 2),
            "num_assets": len(holdings)
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Tool 4: Compare cryptocurrencies
def compare_cryptocurrencies(coin_ids: List[str]) -> str:
    """
    Compare multiple cryptocurrencies side by side.

    Args:
        coin_ids (List[str]): List of CoinGecko coin IDs to compare

    Returns:
        str: JSON formatted string with comparison data
    """
    try:
        comparison = {}
        for coin_id in coin_ids:
            price_data = json.loads(get_coin_price(coin_id))
            if coin_id in price_data:
                comparison[coin_id] = price_data[coin_id]
            time.sleep(0.5)  # Rate limiting
        
        return json.dumps(comparison, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Tool 5: Get market trends
def get_market_trends(limit: int = 5) -> str:
    """
    Analyze market trends from top cryptocurrencies.

    Args:
        limit (int): Number of top coins to analyze. Defaults to 5.

    Returns:
        str: JSON formatted string with trend analysis
    """
    try:
        top_coins = json.loads(get_top_cryptocurrencies(limit))
        trends = {
            "total_market_cap": sum(coin.get("market_cap", 0) for coin in top_coins if isinstance(top_coins, list)),
            "average_24h_change": 0,
            "gainers": [],
            "losers": []
        }
        
        if isinstance(top_coins, list):
            changes = [coin.get("price_change_percentage_24h", 0) for coin in top_coins]
            trends["average_24h_change"] = sum(changes) / len(changes) if changes else 0
            
            for coin in top_coins:
                change = coin.get("price_change_percentage_24h", 0)
                if change > 0:
                    trends["gainers"].append({
                        "id": coin.get("id"),
                        "change": change
                    })
                else:
                    trends["losers"].append({
                        "id": coin.get("id"),
                        "change": change
                    })
        
        return json.dumps(trends, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Initialize agent with multiple tools
agent = Agent(
    agent_name="Multi-Tool-Financial-Agent",
    agent_description="Advanced financial analysis agent with multiple cryptocurrency tools",
    system_prompt="""You are a financial analysis agent with access to multiple cryptocurrency tools.
    You can:
    - Get prices for individual coins
    - Fetch top cryptocurrencies
    - Calculate portfolio values
    - Compare multiple cryptocurrencies
    - Analyze market trends
    
    When a user asks a complex question, use multiple tools together to provide comprehensive answers.
    For example, if asked about portfolio value, use get_coin_price for each holding, then calculate_portfolio_value.
    If asked to compare coins, use compare_cryptocurrencies or get_coin_price multiple times.
    Always provide clear, actionable insights based on the tool results.""",
    max_loops=3,  # Allow multiple tool calls
    model_name="gpt-4o-mini",
    tools=[
        get_coin_price,
        get_top_cryptocurrencies,
        calculate_portfolio_value,
        compare_cryptocurrencies,
        get_market_trends,
    ],
    verbose=True,
)

# Example usage: Complex query requiring multiple tools
response = agent.run(
    "I have 0.5 Bitcoin and 2 Ethereum. Calculate my portfolio value, "
    "then compare Bitcoin and Ethereum, and finally show me the current market trends."
)
print(response)
```

## Example 2: Research Agent with Web Search and Data Processing

This example combines web search, data processing, and analysis tools.

```python
import json
import requests
from typing import List, Dict
from swarms import Agent

# Tool 1: Web search (simulated - replace with actual search API)
def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web for information about a topic.

    Args:
        query (str): Search query
        num_results (int): Number of results to return. Defaults to 5.

    Returns:
        str: JSON formatted string with search results
    """
    # In production, replace with actual search API (Exa, Serper, etc.)
    results = {
        "query": query,
        "results": [
            {"title": f"Result {i} for {query}", "url": f"https://example.com/{i}"}
            for i in range(num_results)
        ]
    }
    return json.dumps(results, indent=2)


# Tool 2: Extract key points from text
def extract_key_points(text: str, num_points: int = 5) -> str:
    """
    Extract key points from a text document.

    Args:
        text (str): Text to analyze
        num_points (int): Number of key points to extract. Defaults to 5.

    Returns:
        str: JSON formatted string with extracted key points
    """
    # Simple extraction (in production, use NLP libraries)
    sentences = text.split('.')
    key_points = {
        "num_points": min(num_points, len(sentences)),
        "points": [s.strip() for s in sentences[:num_points] if s.strip()]
    }
    return json.dumps(key_points, indent=2)


# Tool 3: Summarize information
def summarize_information(points: List[str], max_length: int = 200) -> str:
    """
    Create a concise summary from a list of points.

    Args:
        points (List[str]): List of information points
        max_length (int): Maximum summary length. Defaults to 200.

    Returns:
        str: JSON formatted string with summary
    """
    summary_text = " ".join(points)[:max_length]
    result = {
        "summary": summary_text,
        "num_points_used": len(points),
        "length": len(summary_text)
    }
    return json.dumps(result, indent=2)


# Tool 4: Format research report
def format_research_report(topic: str, findings: List[str], conclusion: str) -> str:
    """
    Format research findings into a structured report.

    Args:
        topic (str): Research topic
        findings (List[str]): List of research findings
        conclusion (str): Conclusion or summary

    Returns:
        str: JSON formatted string with formatted report
    """
    report = {
        "topic": topic,
        "sections": {
            "introduction": f"Research report on {topic}",
            "findings": findings,
            "conclusion": conclusion
        },
        "metadata": {
            "num_findings": len(findings),
            "formatted_at": "2024-01-01"
        }
    }
    return json.dumps(report, indent=2)


# Initialize research agent
research_agent = Agent(
    agent_name="Multi-Tool-Research-Agent",
    agent_description="Research agent with web search and data processing capabilities",
    system_prompt="""You are a research agent with multiple tools for gathering and processing information.
    
    Your workflow:
    1. Use search_web to find information about the topic
    2. Use extract_key_points to identify important information
    3. Use summarize_information to create concise summaries
    4. Use format_research_report to structure your findings
    
    Chain these tools together to create comprehensive research reports.
    Always verify information and cite sources when possible.""",
    max_loops=5,  # Allow multiple tool calls for complex research
    model_name="gpt-4o-mini",
    tools=[
        search_web,
        extract_key_points,
        summarize_information,
        format_research_report,
    ],
    verbose=True,
)

# Example: Complex research task
response = research_agent.run(
    "Research the latest developments in AI and create a formatted report with key findings."
)
print(response)
```

## Example 3: E-commerce Agent with Product Search and Analysis

This example shows an agent using multiple tools for product research and comparison.

```python
import json
from typing import List, Dict
from swarms import Agent

# Tool 1: Search products
def search_products(query: str, category: str = None) -> str:
    """
    Search for products matching a query.

    Args:
        query (str): Product search query
        category (str, optional): Product category filter

    Returns:
        str: JSON formatted string with product results
    """
    # Simulated product data
    products = {
        "query": query,
        "category": category,
        "results": [
            {
                "id": f"prod_{i}",
                "name": f"{query} Product {i}",
                "price": 10.99 + (i * 5),
                "rating": 4.0 + (i * 0.2),
                "in_stock": True
            }
            for i in range(5)
        ]
    }
    return json.dumps(products, indent=2)


# Tool 2: Get product details
def get_product_details(product_id: str) -> str:
    """
    Get detailed information about a specific product.

    Args:
        product_id (str): Product identifier

    Returns:
        str: JSON formatted string with product details
    """
    details = {
        "id": product_id,
        "name": f"Product {product_id}",
        "description": f"Detailed description of {product_id}",
        "price": 29.99,
        "rating": 4.5,
        "reviews_count": 150,
        "specifications": {
            "weight": "1.5 lbs",
            "dimensions": "10x8x2 inches",
            "material": "Premium quality"
        }
    }
    return json.dumps(details, indent=2)


# Tool 3: Compare products
def compare_products(product_ids: List[str]) -> str:
    """
    Compare multiple products side by side.

    Args:
        product_ids (List[str]): List of product IDs to compare

    Returns:
        str: JSON formatted string with comparison data
    """
    comparison = {
        "products": [
            json.loads(get_product_details(pid)) for pid in product_ids
        ],
        "comparison_metrics": {
            "price_range": "Low to High",
            "rating_average": 4.3,
            "total_products": len(product_ids)
        }
    }
    return json.dumps(comparison, indent=2)


# Tool 4: Calculate total cost
def calculate_total_cost(product_ids: List[str], quantities: List[int]) -> str:
    """
    Calculate the total cost for multiple products with quantities.

    Args:
        product_ids (List[str]): List of product IDs
        quantities (List[int]): Corresponding quantities

    Returns:
        str: JSON formatted string with cost breakdown
    """
    total = 0.0
    breakdown = []
    
    for pid, qty in zip(product_ids, quantities):
        details = json.loads(get_product_details(pid))
        price = details.get("price", 0)
        subtotal = price * qty
        total += subtotal
        breakdown.append({
            "product_id": pid,
            "quantity": qty,
            "unit_price": price,
            "subtotal": subtotal
        })
    
    result = {
        "items": breakdown,
        "subtotal": round(total, 2),
        "tax": round(total * 0.08, 2),
        "total": round(total * 1.08, 2)
    }
    return json.dumps(result, indent=2)


# Tool 5: Get recommendations
def get_recommendations(product_id: str, num_recommendations: int = 3) -> str:
    """
    Get product recommendations based on a product.

    Args:
        product_id (str): Base product ID
        num_recommendations (int): Number of recommendations. Defaults to 3.

    Returns:
        str: JSON formatted string with recommendations
    """
    recommendations = {
        "base_product": product_id,
        "recommendations": [
            {
                "id": f"rec_{i}",
                "name": f"Recommended Product {i}",
                "reason": "Similar features",
                "price": 24.99 + (i * 3)
            }
            for i in range(num_recommendations)
        ]
    }
    return json.dumps(recommendations, indent=2)


# Initialize e-commerce agent
ecommerce_agent = Agent(
    agent_name="Multi-Tool-Ecommerce-Agent",
    agent_description="E-commerce agent with product search, comparison, and analysis tools",
    system_prompt="""You are an e-commerce assistant with multiple tools for product research.
    
    You can:
    - Search for products
    - Get detailed product information
    - Compare multiple products
    - Calculate total costs
    - Get product recommendations
    
    When users ask complex questions:
    - Use search_products to find options
    - Use get_product_details for specific products
    - Use compare_products to help users make decisions
    - Use calculate_total_cost for shopping cart calculations
    - Use get_recommendations to suggest alternatives
    
    Always provide helpful, accurate information to help users make informed decisions.""",
    max_loops=4,
    model_name="gpt-4o-mini",
    tools=[
        search_products,
        get_product_details,
        compare_products,
        calculate_total_cost,
        get_recommendations,
    ],
    verbose=True,
)

# Example: Complex e-commerce query
response = ecommerce_agent.run(
    "Search for laptops, get details on the top 3 results, compare them, "
    "and calculate the total cost if I buy 2 of the cheapest one."
)
print(response)
```

## Best Practices for Multiple Tools

### 1. Tool Organization

Group related tools together and give them clear, descriptive names:

```python
# Good: Clear, organized tool names
tools = [
    get_coin_price,           # Financial tools
    get_top_cryptocurrencies,
    calculate_portfolio_value,
    search_web,                # Research tools
    extract_key_points,
    summarize_information,
]
```

### 2. Tool Documentation

Always include comprehensive docstrings so the agent understands when to use each tool:

```python
def my_tool(param: str) -> str:
    """
    Clear description of what the tool does.
    
    Args:
        param (str): Description of parameter
        
    Returns:
        str: Description of return value
        
    Example:
        >>> result = my_tool("example")
    """
    pass
```

### 3. Error Handling

Ensure all tools handle errors gracefully and return consistent JSON:

```python
def robust_tool(input_data: str) -> str:
    """Tool with proper error handling."""
    try:
        # Tool logic
        result = {"status": "success", "data": processed_data}
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
```

### 4. Tool Chaining

Design tools to work together - one tool's output can feed into another:

```python
# Tool 1: Get data
def fetch_data(query: str) -> str:
    return json.dumps({"data": "..."})

# Tool 2: Process data (can use Tool 1's output)
def process_data(data_json: str) -> str:
    data = json.loads(data_json)
    # Process data
    return json.dumps({"processed": "..."})
```

### 5. Rate Limiting

Add rate limiting for API calls to avoid hitting limits:

```python
import time

def api_tool_with_rate_limit(param: str) -> str:
    """Tool with rate limiting."""
    # Make API call
    result = make_api_call(param)
    time.sleep(0.5)  # Rate limiting
    return json.dumps(result)
```

### 6. System Prompt Guidance

Provide clear guidance in the system prompt about tool usage:

```python
system_prompt="""You have access to multiple tools. Here's when to use each:
- Use tool_A for X tasks
- Use tool_B for Y tasks
- Chain tool_A and tool_B together for complex Z tasks
Always explain which tools you're using and why."""
```

## Advanced: Tool Selection Strategies

### Strategy 1: Sequential Tool Chaining

The agent uses tools one after another, with each tool's output informing the next:

```python
# Agent will: search -> extract -> summarize -> format
response = agent.run("Research topic X and create a report")
```

### Strategy 2: Parallel Tool Execution

For independent operations, the agent can use multiple tools in parallel:

```python
# Agent can: get_price(coin1) + get_price(coin2) + get_price(coin3) simultaneously
response = agent.run("Get prices for Bitcoin, Ethereum, and Cardano")
```

### Strategy 3: Conditional Tool Usage

The agent selects tools based on the task:

```python
# Agent decides: if portfolio question -> use calculate_portfolio_value
#                if comparison question -> use compare_cryptocurrencies
response = agent.run("What's the best cryptocurrency to invest in?")
```

## Troubleshooting

### Issue: Agent Not Using Multiple Tools

**Solution**: Increase `max_loops` to allow more tool calls:

```python
agent = Agent(
    max_loops=5,  # Allow up to 5 tool calls
    tools=[...],
)
```

### Issue: Tools Returning Errors

**Solution**: Ensure all tools have proper error handling:

```python
def safe_tool(param: str) -> str:
    try:
        # Tool logic
        return json.dumps({"result": "..."})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

### Issue: Tool Selection Issues

**Solution**: Improve tool docstrings and system prompt:

```python
system_prompt="""Clear instructions on when to use each tool.
Tool A: Use for X
Tool B: Use for Y
Tool C: Use after Tool A for Z"""
```

## Summary

Agents with multiple tools can:

- ✅ Automatically select the right tools for each task
- ✅ Chain tools together for complex workflows
- ✅ Combine different tool categories
- ✅ Handle tool dependencies intelligently

Key takeaways:

1. **Organize tools logically** - Group related tools together
2. **Document thoroughly** - Clear docstrings help the agent choose correctly
3. **Handle errors gracefully** - Always return valid JSON
4. **Design for chaining** - Tools should work well together
5. **Guide with prompts** - System prompts help the agent use tools effectively

For more examples, see:
- [Basic Agent with Tools](./agent_with_tools.md)
- [Agents as Tools](./agents_as_tools.md)
- [Tools Documentation](../tools/tools_examples.md)

