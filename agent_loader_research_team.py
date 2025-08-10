"""
AgentLoader Example: Research Team Collaboration
===============================================

This example demonstrates using the AgentLoader to create a research team
from markdown files and orchestrate them in a sequential workflow.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add local swarms to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.utils.agent_loader import AgentLoader

def create_research_agents():
    """Create markdown files for research team agents"""
    
    market_researcher = """| name | description | model |
|------|-------------|-------|
| market-researcher | Expert in market analysis and competitive intelligence | gpt-4 |

## Focus Areas
- Market size and growth analysis
- Competitive landscape assessment
- Consumer behavior patterns
- Industry trend identification

## Approach
1. Gather comprehensive market data
2. Analyze quantitative and qualitative indicators
3. Identify key market drivers and barriers
4. Evaluate competitive positioning
5. Assess market opportunities and threats

## Output
- Market analysis reports with key metrics
- Competitive intelligence briefings
- Market opportunity assessments
- Consumer behavior insights
"""

    financial_analyst = """| name | description | model |
|------|-------------|-------|
| financial-analyst | Specialist in financial modeling and investment analysis | gpt-4 |

## Focus Areas
- Financial statement analysis
- Valuation modeling techniques
- Investment risk assessment
- Cash flow projections

## Approach
1. Conduct thorough financial analysis
2. Build comprehensive financial models
3. Perform multiple valuation methods
4. Assess financial risks and sensitivities
5. Provide investment recommendations

## Output
- Financial analysis reports
- Valuation models with scenarios
- Investment recommendation memos
- Risk assessment matrices
"""

    industry_expert = """| name | description | model |
|------|-------------|-------|
| industry-expert | Domain specialist with deep industry knowledge | gpt-4 |

## Focus Areas
- Industry structure and dynamics
- Regulatory environment analysis
- Technology trends and disruptions
- Supply chain analysis

## Approach
1. Map industry structure and stakeholders
2. Analyze regulatory framework
3. Identify technology trends
4. Evaluate supply chain dynamics
5. Assess competitive positioning

## Output
- Industry landscape reports
- Regulatory compliance assessments
- Technology trend analysis
- Strategic positioning recommendations
"""

    return {
        "market_researcher.md": market_researcher,
        "financial_analyst.md": financial_analyst,
        "industry_expert.md": industry_expert
    }

def main():
    """Main execution function"""
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create markdown files
        agent_definitions = create_research_agents()
        file_paths = []
        
        for filename, content in agent_definitions.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            file_paths.append(file_path)
        
        # Load agents using AgentLoader
        loader = AgentLoader()
        agents = loader.load_multiple_agents(
            file_paths, 
            max_loops=1,
            verbose=False
        )
        
        print(f"Loaded {len(agents)} agents")
        for i, agent in enumerate(agents):
            print(f"Agent {i}: {agent.agent_name} - LLM: {hasattr(agent, 'llm')}")
        
        # Create sequential workflow
        research_workflow = SequentialWorkflow(
            agents=agents,
            max_loops=1,
        )
        
        # Define research task
        task = """
        Analyze the AI-powered healthcare diagnostics market for a potential $50M investment.
        
        Focus on:
        1. Market size, growth, and key drivers
        2. Competitive landscape and major players
        3. Financial viability and investment metrics
        4. Industry dynamics and regulatory factors
        
        Provide strategic recommendations for market entry.
        """
        
        # Execute workflow
        result = research_workflow.run(task)
        
        return result
        
    finally:
        # Cleanup
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    result = main()
    print("Research Analysis Complete:")
    print("-" * 50)
    print(result)