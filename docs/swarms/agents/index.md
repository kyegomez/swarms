# Agents Introduction


An agent in swarms is basically 4 elements added together:

`agent = LLM + Tools + RAG + Loop`

The Agent class is the core component of the Swarms framework, designed to create intelligent, autonomous AI agents capable of handling complex tasks through multi-modal processing, tool integration, and structured outputs. This comprehensive guide covers all aspects of the Agent class, from basic setup to advanced features.


## Prerequisites & Installation

### System Requirements

- Python 3.7+

- OpenAI API key (for GPT models)

- Anthropic API key (for Claude models)

### Installation

```bash
pip3 install -U swarms
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
WORKSPACE_DIR="agent_workspace"
```

## Basic Agent Configuration

### Core Agent Structure

The Agent class provides a comprehensive set of parameters for customization:

```python
from swarms import Agent

# Basic agent initialization
agent = Agent(
    agent_name="MyAgent",
    agent_description="A specialized AI agent for specific tasks",
    system_prompt="You are a helpful assistant...",
    model_name="gpt-4o-mini",
    max_loops=1,
    max_tokens=4096,
    temperature=0.7,
    output_type="str",
    safety_prompt_on=True
)
```

### Key Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `agent_name` | str | Unique identifier for the agent | Required |
| `agent_description` | str | Detailed description of capabilities | Required |
| `system_prompt` | str | Core instructions defining behavior | Required |
| `model_name` | str | AI model to use | "gpt-4o-mini" |
| `max_loops` | int | Maximum execution loops | 1 |
| `max_tokens` | int | Maximum response tokens | 4096 |
| `temperature` | float | Response creativity (0-1) | 0.7 |
| `output_type` | str | Response format type | "str" |
| `multi_modal` | bool | Enable image processing | False |
| `safety_prompt_on` | bool | Enable safety checks | True |

### Simple Example

```python
from swarms import Agent

# Create a basic financial advisor agent
financial_agent = Agent(
    agent_name="Financial-Advisor",
    agent_description="Personal finance and investment advisor",
    system_prompt="""You are an expert financial advisor with deep knowledge of:
    - Investment strategies and portfolio management
    - Risk assessment and mitigation
    - Market analysis and trends
    - Financial planning and budgeting
    
    Provide clear, actionable advice while considering risk tolerance.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    temperature=0.3,
    output_type="str"
)

# Run the agent
response = financial_agent.run("What are the best investment strategies for a 30-year-old?")
print(response)
```

## Multi-Modal Capabilities

### Image Processing

The Agent class supports comprehensive image analysis through vision-enabled models:

```python
from swarms import Agent

# Create a vision-enabled agent
vision_agent = Agent(
    agent_name="Vision-Analyst",
    agent_description="Advanced image analysis and quality control agent",
    system_prompt="""You are an expert image analyst capable of:
    - Detailed visual inspection and quality assessment
    - Object detection and classification
    - Scene understanding and context analysis
    - Defect identification and reporting
    
    Provide comprehensive analysis with specific observations.""",
    model_name="gpt-4o-mini",  # Vision-enabled model
    multi_modal=True,  # Enable multi-modal processing
    max_loops=1,
    output_type="str"
)

# Analyze a single image
response = vision_agent.run(
    task="Analyze this image for quality control purposes",
    img="path/to/image.jpg"
)

# Process multiple images
response = vision_agent.run(
    task="Compare these images and identify differences",
    imgs=["image1.jpg", "image2.jpg", "image3.jpg"],
    summarize_multiple_images=True
)
```

### Supported Image Formats

| Format | Description | Max Size |
|--------|-------------|----------|
| JPEG/JPG | Standard compressed format | 20MB |
| PNG | Lossless with transparency | 20MB |
| GIF | Animated (first frame only) | 20MB |
| WebP | Modern efficient format | 20MB |

### Quality Control Example

```python
from swarms import Agent
from swarms.prompts.logistics import Quality_Control_Agent_Prompt

def security_analysis(danger_level: str) -> str:
    """Analyze security danger level and return appropriate response."""
    danger_responses = {
        "low": "No immediate danger detected",
        "medium": "Moderate security concern identified",
        "high": "Critical security threat detected",
        None: "No danger level assessment available"
    }
    return danger_responses.get(danger_level, "Unknown danger level")

# Quality control agent with tool integration
quality_agent = Agent(
    agent_name="Quality-Control-Agent",
    agent_description="Advanced quality control and security analysis agent",
    system_prompt=f"""
    {Quality_Control_Agent_Prompt}
    
    You have access to security analysis tools. When analyzing images:
    1. Identify potential safety hazards
    2. Assess quality standards compliance
    3. Determine appropriate danger levels (low, medium, high)
    4. Use the security_analysis function for threat assessment
    """,
    model_name="gpt-4o-mini",
    multi_modal=True,
    max_loops=1,
    tools=[security_analysis]
)

# Analyze factory image
response = quality_agent.run(
    task="Analyze this factory image for safety and quality issues",
    img="factory_floor.jpg"
)
```

## Tool Integration

### Creating Custom Tools

Tools are Python functions that extend your agent's capabilities:

```python
import json
import requests
from typing import Optional, Dict, Any

def get_weather_data(city: str, country: Optional[str] = None) -> str:
    """
    Get current weather data for a specified city.
    
    Args:
        city (str): The city name
        country (Optional[str]): Country code (e.g., 'US', 'UK')
    
    Returns:
        str: JSON formatted weather data
    
    Example:
        >>> weather = get_weather_data("San Francisco", "US")
        >>> print(weather)
        {"temperature": 18, "condition": "partly cloudy", ...}
    """
    try:
        # API call logic here
        weather_data = {
            "city": city,
            "country": country,
            "temperature": 18,
            "condition": "partly cloudy",
            "humidity": 65,
            "wind_speed": 12
        }
        return json.dumps(weather_data, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"Weather API error: {str(e)}"})

def calculate_portfolio_metrics(prices: list, weights: list) -> str:
    """
    Calculate portfolio performance metrics.
    
    Args:
        prices (list): List of asset prices
        weights (list): List of portfolio weights
    
    Returns:
        str: JSON formatted portfolio metrics
    """
    try:
        # Portfolio calculation logic
        portfolio_value = sum(p * w for p, w in zip(prices, weights))
        metrics = {
            "total_value": portfolio_value,
            "weighted_average": portfolio_value / sum(weights),
            "asset_count": len(prices)
        }
        return json.dumps(metrics, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"Calculation error: {str(e)}"})
```

### Tool Integration Example

```python
from swarms import Agent

# Create agent with custom tools
multi_tool_agent = Agent(
    agent_name="Multi-Tool-Assistant",
    agent_description="Versatile assistant with weather and financial tools",
    system_prompt="""You are a versatile assistant with access to:
    - Weather data retrieval for any city
    - Portfolio analysis and financial calculations
    
    Use these tools to provide comprehensive assistance.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    tools=[get_weather_data, calculate_portfolio_metrics]
)

# Use the agent with tools
response = multi_tool_agent.run(
    "What's the weather in New York and calculate metrics for a portfolio with prices [100, 150, 200] and weights [0.3, 0.4, 0.3]?"
)
```

### API Integration Tools

```python
import requests
import json
from typing import List

def get_cryptocurrency_price(coin_id: str, vs_currency: str = "usd") -> str:
    """Get current cryptocurrency price from CoinGecko API."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"API error: {str(e)}"})

def get_top_cryptocurrencies(limit: int = 10) -> str:
    """Get top cryptocurrencies by market cap."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"API error: {str(e)}"})

# Crypto analysis agent
crypto_agent = Agent(
    agent_name="Crypto-Analysis-Agent",
    agent_description="Cryptocurrency market analysis and price tracking agent",
    system_prompt="""You are a cryptocurrency analysis expert with access to:
    - Real-time price data for any cryptocurrency
    - Market capitalization rankings
    - Trading volume and price change data
    
    Provide insightful market analysis and investment guidance.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    tools=[get_cryptocurrency_price, get_top_cryptocurrencies]
)

# Analyze crypto market
response = crypto_agent.run("Analyze the current Bitcoin price and show me the top 5 cryptocurrencies")
```

## Structured Outputs

### Function Schema Definition

Define structured outputs using OpenAI's function calling format:

```python
from swarms import Agent

# Define function schemas for structured outputs
stock_analysis_schema = {
    "type": "function",
    "function": {
        "name": "analyze_stock_performance",
        "description": "Analyze stock performance with detailed metrics",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["technical", "fundamental", "comprehensive"],
                    "description": "Type of analysis to perform"
                },
                "time_period": {
                    "type": "string",
                    "enum": ["1d", "1w", "1m", "3m", "1y"],
                    "description": "Time period for analysis"
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["price", "volume", "pe_ratio", "market_cap", "volatility"]
                    },
                    "description": "Metrics to include in analysis"
                }
            },
            "required": ["ticker", "analysis_type"]
        }
    }
}

portfolio_optimization_schema = {
    "type": "function",
    "function": {
        "name": "optimize_portfolio",
        "description": "Optimize portfolio allocation based on risk and return",
        "parameters": {
            "type": "object",
            "properties": {
                "assets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "current_weight": {"type": "number"},
                            "expected_return": {"type": "number"},
                            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
                        },
                        "required": ["symbol", "current_weight"]
                    }
                },
                "risk_tolerance": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive"]
                },
                "investment_horizon": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Investment time horizon in years"
                }
            },
            "required": ["assets", "risk_tolerance"]
        }
    }
}

# Create agent with structured outputs
structured_agent = Agent(
    agent_name="Structured-Financial-Agent",
    agent_description="Financial analysis agent with structured output capabilities",
    system_prompt="""You are a financial analysis expert that provides structured outputs.
    Use the provided function schemas to format your responses consistently.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    tools_list_dictionary=[stock_analysis_schema, portfolio_optimization_schema]
)

# Generate structured analysis
response = structured_agent.run(
    "Analyze Apple stock (AAPL) performance with comprehensive analysis for the last 3 months"
)
```

## Advanced Features

### Dynamic Temperature Control

```python
from swarms import Agent

# Agent with dynamic temperature adjustment
adaptive_agent = Agent(
    agent_name="Adaptive-Response-Agent",
    agent_description="Agent that adjusts response creativity based on context",
    system_prompt="You are an adaptive AI that adjusts your response style based on the task complexity.",
    model_name="gpt-4o-mini",
    dynamic_temperature_enabled=True,  # Enable adaptive temperature
    max_loops=1,
    output_type="str"
)
```

### Output Type Configurations

```python
# Different output type examples
json_agent = Agent(
    agent_name="JSON-Agent",
    system_prompt="Always respond in valid JSON format",
    output_type="json"
)

streaming_agent = Agent(
    agent_name="Streaming-Agent", 
    system_prompt="Provide detailed streaming responses",
    output_type="str-all-except-first"
)

final_only_agent = Agent(
    agent_name="Final-Only-Agent",
    system_prompt="Provide only the final result",
    output_type="final"
)
```


### Performance Optimization

```python
from swarms import Agent
import time

# Optimized agent configuration
optimized_agent = Agent(
    agent_name="Optimized-Agent",
    agent_description="Performance-optimized agent configuration",
    system_prompt="You are an efficient AI assistant optimized for performance.",
    model_name="gpt-4o-mini",  # Faster model
    max_loops=1,  # Minimize loops
    max_tokens=2048,  # Reasonable token limit
    temperature=0.5,  # Balanced creativity
    output_type="str"
)

# Batch processing example
def process_tasks_batch(agent, tasks, batch_size=5):
    """Process multiple tasks efficiently."""
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = []
        
        for task in batch:
            start_time = time.time()
            result = agent.run(task)
            execution_time = time.time() - start_time
            
            batch_results.append({
                "task": task,
                "result": result,
                "execution_time": execution_time
            })
        
        results.extend(batch_results)
        time.sleep(1)  # Rate limiting
    
    return results
```

## Complete Examples

### Multi-Modal Quality Control System

```python
from swarms import Agent
from swarms.prompts.logistics import Quality_Control_Agent_Prompt

def security_analysis(danger_level: str) -> str:
    """Analyze security danger level and return appropriate response."""
    responses = {
        "low": "✅ No immediate danger detected - Safe to proceed",
        "medium": "⚠️ Moderate security concern - Requires attention",
        "high": "🚨 Critical security threat - Immediate action required",
        None: "❓ No danger level assessment available"
    }
    return responses.get(danger_level, "Unknown danger level")

def quality_assessment(quality_score: int) -> str:
    """Assess quality based on numerical score (1-10)."""
    if quality_score >= 8:
        return "✅ Excellent quality - Meets all standards"
    elif quality_score >= 6:
        return "⚠️ Good quality - Minor improvements needed"
    elif quality_score >= 4:
        return "❌ Poor quality - Significant issues identified"
    else:
        return "🚨 Critical quality failure - Immediate attention required"

# Advanced quality control agent
quality_control_system = Agent(
    agent_name="Advanced-Quality-Control-System",
    agent_description="Comprehensive quality control and security analysis system",
    system_prompt=f"""
    {Quality_Control_Agent_Prompt}
    
    You are an advanced quality control system with the following capabilities:
    
    1. Visual Inspection: Analyze images for defects, compliance, and safety
    2. Security Assessment: Identify potential security threats and hazards
    3. Quality Scoring: Provide numerical quality ratings (1-10 scale)
    4. Detailed Reporting: Generate comprehensive analysis reports
    
    When analyzing images:
    - Identify specific defects or issues
    - Assess compliance with safety standards
    - Determine appropriate danger levels (low, medium, high)
    - Provide quality scores and recommendations
    - Use available tools for detailed analysis
    
    Always provide specific, actionable feedback.
    """,
    model_name="gpt-4o-mini",
    multi_modal=True,
    max_loops=1,
    tools=[security_analysis, quality_assessment],
    output_type="str"
)

# Process factory images
factory_images = ["factory_floor.jpg", "assembly_line.jpg", "safety_equipment.jpg"]

for image in factory_images:
    print(f"\n--- Analyzing {image} ---")
    response = quality_control_system.run(
        task=f"Perform comprehensive quality control analysis of this image. Assess safety, quality, and provide specific recommendations.",
        img=image
    )
    print(response)
```

### Advanced Financial Analysis Agent

```python
from swarms import Agent
import json
import requests

def get_market_data(symbol: str, period: str = "1y") -> str:
    """Get comprehensive market data for a symbol."""
    # Simulated market data (replace with real API)
    market_data = {
        "symbol": symbol,
        "current_price": 150.25,
        "change_percent": 2.5,
        "volume": 1000000,
        "market_cap": 2500000000,
        "pe_ratio": 25.5,
        "dividend_yield": 1.8,
        "52_week_high": 180.50,
        "52_week_low": 120.30
    }
    return json.dumps(market_data, indent=2)

def calculate_risk_metrics(prices: list, benchmark_prices: list) -> str:
    """Calculate risk metrics for a portfolio."""
    import numpy as np
    
    try:
        returns = np.diff(prices) / prices[:-1]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
        
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        max_drawdown = np.max(np.maximum.accumulate(prices) - prices) / np.max(prices)
        
        beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        
        risk_metrics = {
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "beta": float(beta)
        }
        
        return json.dumps(risk_metrics, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"Risk calculation error: {str(e)}"})

# Financial analysis schemas
financial_analysis_schema = {
    "type": "function",
    "function": {
        "name": "comprehensive_financial_analysis",
        "description": "Perform comprehensive financial analysis with structured output",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_summary": {
                    "type": "object",
                    "properties": {
                        "overall_rating": {"type": "string", "enum": ["buy", "hold", "sell"]},
                        "confidence_level": {"type": "number", "minimum": 0, "maximum": 100},
                        "key_strengths": {"type": "array", "items": {"type": "string"}},
                        "key_concerns": {"type": "array", "items": {"type": "string"}},
                        "price_target": {"type": "number"},
                        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
                    }
                },
                "technical_analysis": {
                    "type": "object",
                    "properties": {
                        "trend_direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                        "support_levels": {"type": "array", "items": {"type": "number"}},
                        "resistance_levels": {"type": "array", "items": {"type": "number"}},
                        "momentum_indicators": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["analysis_summary", "technical_analysis"]
        }
    }
}

# Advanced financial agent
financial_analyst = Agent(
    agent_name="Advanced-Financial-Analyst",
    agent_description="Comprehensive financial analysis and investment advisory agent",
    system_prompt="""You are an expert financial analyst with advanced capabilities in:
    
    - Fundamental analysis and valuation
    - Technical analysis and chart patterns
    - Risk assessment and portfolio optimization
    - Market sentiment analysis
    - Economic indicator interpretation
    
    Your analysis should be:
    - Data-driven and objective
    - Risk-aware and practical
    - Clearly structured and actionable
    - Compliant with financial regulations
    
    Use available tools to gather market data and calculate risk metrics.
    Provide structured outputs using the defined schemas.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    tools=[get_market_data, calculate_risk_metrics],
    tools_list_dictionary=[financial_analysis_schema],
    output_type="json"
)

# Comprehensive financial analysis
analysis_response = financial_analyst.run(
    "Perform a comprehensive analysis of Apple Inc. (AAPL) including technical and fundamental analysis with structured recommendations"
)

print(json.dumps(json.loads(analysis_response), indent=2))
```

### Multi-Agent Collaboration System

```python
from swarms import Agent
import json

# Specialized agents for different tasks
research_agent = Agent(
    agent_name="Research-Specialist",
    agent_description="Market research and data analysis specialist",
    system_prompt="You are a market research expert specializing in data collection and analysis.",
    model_name="gpt-4o-mini",
    max_loops=1,
    temperature=0.3
)

strategy_agent = Agent(
    agent_name="Strategy-Advisor", 
    agent_description="Strategic planning and recommendation specialist",
    system_prompt="You are a strategic advisor providing high-level recommendations based on research.",
    model_name="gpt-4o-mini",
    max_loops=1,
    temperature=0.5
)

execution_agent = Agent(
    agent_name="Execution-Planner",
    agent_description="Implementation and execution planning specialist", 
    system_prompt="You are an execution expert creating detailed implementation plans.",
    model_name="gpt-4o-mini",
    max_loops=1,
    temperature=0.4
)

def collaborative_analysis(topic: str):
    """Perform collaborative analysis using multiple specialized agents."""
    
    # Step 1: Research Phase
    research_task = f"Conduct comprehensive research on {topic}. Provide key findings, market data, and trends."
    research_results = research_agent.run(research_task)
    
    # Step 2: Strategy Phase
    strategy_task = f"Based on this research: {research_results}\n\nDevelop strategic recommendations for {topic}."
    strategy_results = strategy_agent.run(strategy_task)
    
    # Step 3: Execution Phase
    execution_task = f"Create a detailed implementation plan based on:\nResearch: {research_results}\nStrategy: {strategy_results}"
    execution_results = execution_agent.run(execution_task)
    
    return {
        "research": research_results,
        "strategy": strategy_results,
        "execution": execution_results
    }

# Example: Collaborative investment analysis
investment_analysis = collaborative_analysis("renewable energy sector investment opportunities")

for phase, results in investment_analysis.items():
    print(f"\n=== {phase.upper()} PHASE ===")
    print(results)
```

## Support and Resources

Join our community of agent engineers and researchers for technical support, cutting-edge updates, and exclusive access to world-class agent engineering insights!

| Platform | Description | Link |
|----------|-------------|------|
| 📚 Documentation | Official documentation and guides | [docs.swarms.world](https://docs.swarms.world) |
| 📝 Blog | Latest updates and technical articles | [Medium](https://medium.com/@kyeg) |
| 💬 Discord | Live chat and community support | [Join Discord](https://discord.gg/EamjgSaEQf) |
| 🐦 Twitter | Latest news and announcements | [@kyegomez](https://twitter.com/kyegomez) |
| 👥 LinkedIn | Professional network and updates | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| 📺 YouTube | Tutorials and demos | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) |
| 🎫 Events | Join our community events | [Sign up here](https://lu.ma/5p2jnc2v) |
| 🚀 Onboarding Session | Get onboarded with Kye Gomez, creator and lead maintainer of Swarms | [Book Session](https://cal.com/swarms/swarms-onboarding-session) |

### Getting Help

If you encounter issues or need assistance:

1. **Check the Documentation**: Start with the official docs for comprehensive guides
2. **Search Issues**: Look through existing GitHub issues for similar problems
3. **Join Discord**: Get real-time help from the community
4. **Create an Issue**: Report bugs or request features on GitHub
5. **Follow Updates**: Stay informed about new releases and improvements

### Contributing

We welcome contributions! Here's how to get involved:

| Contribution Type        | Description                                      |
|-------------------------|--------------------------------------------------|
| **Report Bugs**         | Help us improve by reporting issues              |
| **Suggest Features**    | Share your ideas for new capabilities            |
| **Submit Code**         | Contribute improvements and new features         |
| **Improve Documentation** | Help make our docs better                      |
| **Share Examples**      | Show how you're using Swarms in your projects    |

---

*This guide covers the essential aspects of the Swarms Agent class. For the most up-to-date information and advanced features, please refer to the official documentation and community resources.*