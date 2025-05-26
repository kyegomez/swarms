from dotenv import load_dotenv

from swarms import Agent
from swarms.utils.vllm_wrapper import VLLM

load_dotenv()

# Define custom system prompt for crypto risk analysis
CRYPTO_RISK_ANALYSIS_PROMPT = """
You are a cryptocurrency risk analysis expert. Your role is to:

1. Analyze market risks:
   - Volatility assessment
   - Market sentiment analysis
   - Trading volume patterns
   - Price trend evaluation

2. Evaluate technical risks:
   - Network security
   - Protocol vulnerabilities
   - Smart contract risks
   - Technical scalability

3. Consider regulatory risks:
   - Current regulations
   - Potential regulatory changes
   - Compliance requirements
   - Geographic restrictions

4. Assess fundamental risks:
   - Team background
   - Project development status
   - Competition analysis
   - Use case viability

Provide detailed, balanced analysis with both risks and potential mitigations.
Base your analysis on established crypto market principles and current market conditions.
"""

model = VLLM(model_name="meta-llama/Llama-4-Maverick-17B-128E")

# Initialize the agent with custom prompt
agent = Agent(
    agent_name="Crypto-Risk-Analysis-Agent",
    agent_description="Agent for analyzing risks in cryptocurrency investments",
    system_prompt=CRYPTO_RISK_ANALYSIS_PROMPT,
    max_loops=1,
    llm=model,
)

print(
    agent.run(
        "Conduct a risk analysis of the top cryptocurrencies. Think for 2 loops internally"
    )
)
