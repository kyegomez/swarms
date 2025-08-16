"""
Simple AgentLoader Demo - Claude Code Format
=============================================

A comprehensive demonstration of the AgentLoader using the Claude Code 
sub-agent YAML frontmatter format.

This example shows:
1. Creating agents using Claude Code YAML frontmatter format
2. Loading agents from markdown files with YAML frontmatter
3. Using loaded agents in multi-agent workflows
4. Demonstrating different agent configurations
"""

import os
import tempfile
from pathlib import Path
import sys

# Add local swarms to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.utils.agent_loader import AgentLoader, load_agents_from_markdown

def create_markdown_agent_files():
    """Create markdown files demonstrating Claude Code YAML frontmatter format"""
    
    # Claude Code YAML frontmatter format
    agent_files = {
        "market_researcher.md": """---
name: market-researcher
description: Expert in market analysis and competitive intelligence
model_name: gpt-4
temperature: 0.2
max_loops: 2
mcp_url: http://example.com/market-data
---

You are a market research specialist with deep expertise in analyzing market dynamics and competitive landscapes.

Your core responsibilities include:
- Conducting comprehensive market size and growth analysis
- Performing detailed competitive landscape assessments
- Analyzing consumer behavior patterns and preferences  
- Identifying emerging industry trends and opportunities

Methodology:
1. Gather comprehensive quantitative and qualitative market data
2. Analyze key market drivers, barriers, and success factors
3. Evaluate competitive positioning and market share dynamics
4. Assess market opportunities, threats, and entry strategies
5. Provide actionable insights with data-driven recommendations

Always provide detailed market analysis reports with specific metrics, growth projections, and strategic recommendations for market entry or expansion.
""",

        "financial_analyst.md": """---
name: financial-analyst
description: Specialist in financial modeling and investment analysis
model_name: gpt-4
temperature: 0.1
max_loops: 3
---

You are a financial analysis expert specializing in investment evaluation and financial modeling.

Your areas of expertise include:
- Financial statement analysis and ratio interpretation
- DCF modeling and valuation techniques (DCF, comparable company analysis, precedent transactions)
- Investment risk assessment and sensitivity analysis
- Cash flow projections and working capital analysis

Analytical approach:
1. Conduct thorough financial statement analysis
2. Build comprehensive financial models with multiple scenarios
3. Perform detailed valuation using multiple methodologies
4. Assess financial risks and conduct sensitivity analysis
5. Generate investment recommendations with clear rationale

Provide detailed financial reports with valuation models, risk assessments, and investment recommendations supported by quantitative analysis.
""",

        "industry_expert.md": """---
name: industry-expert
description: Domain specialist with deep industry knowledge and regulatory expertise
model_name: gpt-4
temperature: 0.3
max_loops: 2
---

You are an industry analysis expert with comprehensive knowledge of market structures, regulatory environments, and technology trends.

Your specialization areas:
- Industry structure analysis and value chain mapping
- Regulatory environment assessment and compliance requirements
- Technology trends identification and disruption analysis
- Supply chain dynamics and operational considerations

Research methodology:
1. Map industry structure, key players, and stakeholder relationships
2. Analyze current and emerging regulatory framework
3. Identify technology trends and potential market disruptions
4. Evaluate supply chain dynamics and operational requirements
5. Assess competitive positioning and strategic opportunities

Generate comprehensive industry landscape reports with regulatory insights, technology trend analysis, and strategic recommendations for market positioning.
"""
        "risk_analyst.md": """---
name: risk-analyst
description: Specialist in investment risk assessment and mitigation strategies
model_name: gpt-4
temperature: 0.15
max_loops: 2
---

You are a Risk Analyst specializing in comprehensive investment risk assessment and portfolio management.

Your core competencies include:
- Conducting detailed investment risk evaluation and categorization
- Implementing sophisticated portfolio risk management strategies
- Ensuring regulatory compliance and conducting compliance assessments
- Performing advanced scenario analysis and stress testing methodologies

Analytical framework:
1. Systematically identify and categorize all investment risks
2. Quantify risk exposure using advanced statistical methods and models
3. Develop comprehensive risk mitigation strategies and frameworks
4. Conduct rigorous scenario analysis and stress testing procedures
5. Provide actionable risk management recommendations with implementation roadmaps

Deliver comprehensive risk assessment reports with quantitative analysis, compliance guidelines, and strategic risk management recommendations.
"""
    }
    
    temp_files = []
    
    # Create Claude Code format files
    for filename, content in agent_files.items():
        temp_file = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        temp_files.append(temp_file)
    
    return temp_files

def main():
    """Main execution function demonstrating AgentLoader with Claude Code format"""
    
    print("AgentLoader Demo - Claude Code YAML Frontmatter Format")
    print("=" * 60)
    
    # Create markdown files demonstrating both formats
    print("\n1. Creating markdown files...")
    temp_files = create_markdown_agent_files()
    
    try:
        # Load agents using AgentLoader
        print("\n2. Loading agents using AgentLoader...")
        agents = load_agents_from_markdown(temp_files)
        
        print(f"   Successfully loaded {len(agents)} agents:")
        for agent in agents:
            temp_attr = getattr(agent, 'temperature', 'default')
            max_loops = getattr(agent, 'max_loops', 1)
            print(f"   - {agent.agent_name} (temp: {temp_attr}, loops: {max_loops})")
        
        # Demonstrate individual agent configuration
        print("\n3. Agent Configuration Details:")
        for i, agent in enumerate(agents, 1):
            print(f"   Agent {i}: {agent.agent_name}")
            print(f"   Model: {getattr(agent, 'model_name', 'default')}")
            print(f"   System prompt preview: {agent.system_prompt[:100]}...")
            print()
        
        # Create sequential workflow with loaded agents
        print("4. Creating sequential workflow...")
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
        5. Risk assessment and mitigation strategies
        
        Provide comprehensive strategic recommendations for market entry.
        """
        
        print("5. Executing research workflow...")
        print("=" * 50)
        
        # Note: In a real scenario, this would execute the workflow
        # For demo purposes, we'll show the task distribution
        print(f"Task distributed to {len(agents)} specialized agents:")
        for i, agent in enumerate(agents, 1):
            print(f"   Agent {i} ({agent.agent_name}): Ready to process")
        
        print(f"\nTask preview: {task[:150]}...")
        print("\n[Demo mode - actual workflow execution would call LLM APIs]")
        
        print("\nDemo Summary:")
        print("-" * 50)
        print("✓ Successfully loaded agents using Claude Code YAML frontmatter format")
        print("✓ Agents configured with different temperatures and max_loops from YAML")
        print("✓ Multi-agent workflow created with specialized investment analysis agents")
        print("✓ Workflow ready for comprehensive market analysis execution")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup temporary files
        print("\n6. Cleaning up temporary files...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"   Removed: {os.path.basename(temp_file)}")
            except OSError:
                pass

if __name__ == "__main__":
    main()