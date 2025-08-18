# Mixture of Agents Example

The Mixture of Agents (MoA) is a sophisticated multi-agent architecture that implements parallel processing with iterative refinement. This approach processes multiple specialized agents simultaneously, concatenates their outputs, and then performs multiple parallel runs to achieve consensus or enhanced results.

## How It Works

1. **Parallel Processing**: Multiple agents work simultaneously on the same input
2. **Output Concatenation**: Results from all agents are combined into a unified response
3. **Iterative Refinement**: The process repeats for `n` layers/iterations to improve quality
4. **Consensus Building**: Multiple runs help achieve more reliable and comprehensive outputs

This architecture is particularly effective for complex tasks that benefit from diverse perspectives and iterative improvement, such as financial analysis, risk assessment, and multi-faceted problem solving.

![Mixture of Agents](https://files.readme.io/ddb138e-moa-3layer.png)


## Installation

Install the swarms package using pip:

```bash
pip install -U swarms
```

## Basic Setup

1. First, set up your environment variables:

```python
WORKSPACE_DIR="agent_workspace"
ANTHROPIC_API_KEY=""
```

## Code

```python
from swarms import Agent, MixtureOfAgents

# Agent 1: Risk Metrics Calculator
risk_metrics_agent = Agent(
    agent_name="Risk-Metrics-Calculator",
    agent_description="Calculates key risk metrics like VaR, Sharpe ratio, and volatility",
    system_prompt="""You are a risk metrics specialist. Calculate and explain:
    - Value at Risk (VaR)
    - Sharpe ratio
    - Volatility
    - Maximum drawdown
    - Beta coefficient
    
    Provide clear, numerical results with brief explanations.""",
    max_loops=1,
    # model_name="gpt-4o-mini",
    random_model_enabled=True,
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_tokens=4096,
)

# Agent 2: Portfolio Risk Analyzer
portfolio_risk_agent = Agent(
    agent_name="Portfolio-Risk-Analyzer",
    agent_description="Analyzes portfolio diversification and concentration risk",
    system_prompt="""You are a portfolio risk analyst. Focus on:
    - Portfolio diversification analysis
    - Concentration risk assessment
    - Correlation analysis
    - Sector/asset allocation risk
    - Liquidity risk evaluation
    
    Provide actionable insights for risk reduction.""",
    max_loops=1,
    # model_name="gpt-4o-mini",
    random_model_enabled=True,
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_tokens=4096,
)

# Agent 3: Market Risk Monitor
market_risk_agent = Agent(
    agent_name="Market-Risk-Monitor",
    agent_description="Monitors market conditions and identifies risk factors",
    system_prompt="""You are a market risk monitor. Identify and assess:
    - Market volatility trends
    - Economic risk factors
    - Geopolitical risks
    - Interest rate risks
    - Currency risks
    
    Provide current risk alerts and trends.""",
    max_loops=1,
    # model_name="gpt-4o-mini",
    random_model_enabled=True,
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_tokens=4096,
)


swarm = MixtureOfAgents(
    agents=[
        risk_metrics_agent,
        portfolio_risk_agent,
        market_risk_agent,
    ],
    layers=1,
    max_loops=1,
    output_type="final",
)


out = swarm.run(
    "Calculate VaR and Sharpe ratio for a portfolio with 15% annual return and 20% volatility"
)

print(out)
```

## Support and Community

If you're facing issues or want to learn more, check out the following resources to join our Discord, stay updated on Twitter, and watch tutorials on YouTube!

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 📝 Blog | [Medium](https://medium.com/@kyeg) | Latest updates and technical articles |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@kyegomez](https://twitter.com/kyegomez) | Latest news and announcements |
| 👥 LinkedIn | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Professional network and updates |
| 📺 YouTube | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) | Tutorials and demos |
| 🎫 Events | [Sign up here](https://lu.ma/5p2jnc2v) | Join our community events |

