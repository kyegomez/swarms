"""
Simple AgentLoader Demo
=======================

A working demonstration of how to create agents from markdown-like definitions
and use them in workflows.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add local swarms to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

def create_agents_from_configs():
    """Create agents from configuration dictionaries (simulating markdown parsing)"""
    
    # These would normally come from parsing markdown files
    agent_configs = [
        {
            "name": "market-researcher",
            "description": "Expert in market analysis and competitive intelligence",
            "system_prompt": """You are a market research specialist. Your expertise includes:

Focus Areas:
- Market size and growth analysis
- Competitive landscape assessment  
- Consumer behavior patterns
- Industry trend identification

Approach:
1. Gather comprehensive market data
2. Analyze quantitative and qualitative indicators
3. Identify key market drivers and barriers
4. Evaluate competitive positioning
5. Assess market opportunities and threats

Provide detailed market analysis reports with key metrics and actionable insights.""",
            "model": "gpt-4"
        },
        {
            "name": "financial-analyst", 
            "description": "Specialist in financial modeling and investment analysis",
            "system_prompt": """You are a financial analysis expert. Your responsibilities include:

Focus Areas:
- Financial statement analysis
- Valuation modeling techniques
- Investment risk assessment
- Cash flow projections

Approach:
1. Conduct thorough financial analysis
2. Build comprehensive financial models
3. Perform multiple valuation methods
4. Assess financial risks and sensitivities
5. Provide investment recommendations

Generate detailed financial reports with valuation models and risk assessments.""",
            "model": "gpt-4"
        },
        {
            "name": "industry-expert",
            "description": "Domain specialist with deep industry knowledge", 
            "system_prompt": """You are an industry analysis expert. Your focus areas include:

Focus Areas:
- Industry structure and dynamics
- Regulatory environment analysis
- Technology trends and disruptions
- Supply chain analysis

Approach:
1. Map industry structure and stakeholders
2. Analyze regulatory framework
3. Identify technology trends
4. Evaluate supply chain dynamics
5. Assess competitive positioning

Provide comprehensive industry landscape reports with strategic recommendations.""",
            "model": "gpt-4"
        }
    ]
    
    agents = []
    for config in agent_configs:
        agent = Agent(
            agent_name=config["name"],
            system_prompt=config["system_prompt"],
            model_name=config["model"],
            max_loops=1,
            verbose=False
        )
        agents.append(agent)
        print(f"Created agent: {agent.agent_name}")
    
    return agents

def main():
    """Main execution function"""
    
    # Create agents
    agents = create_agents_from_configs()
    
    # Create sequential workflow
    research_workflow = SequentialWorkflow(
        agents=agents,
        max_loops=1,
    )
    
    # Define research task
    task = """
    Analyze the AI-powered healthcare diagnostics market for a potential $50M investment.
    
    Focus on:
    1. Market size, growth projections, and key drivers
    2. Competitive landscape and major players  
    3. Financial viability and investment attractiveness
    4. Industry dynamics and regulatory considerations
    
    Provide strategic recommendations for market entry.
    """
    
    print("Executing research workflow...")
    print("=" * 50)
    
    # Execute workflow
    result = research_workflow.run(task)
    
    print("\nResearch Analysis Complete:")
    print("-" * 50)
    print(result)

if __name__ == "__main__":
    main()