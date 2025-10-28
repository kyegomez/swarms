from swarms import Agent
from swarms.structs.aop import (
    AOP,
)

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert in research, data collection, and information gathering",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a research specialist. Your role is to:
    1. Gather comprehensive information on any given topic
    2. Analyze data from multiple sources
    3. Provide well-structured research findings
    4. Cite sources and maintain accuracy
    5. Present findings in a clear, organized manner
    
    Always provide detailed, factual information with proper context.""",
)

analysis_agent = Agent(
    agent_name="Analysis-Agent",
    agent_description="Expert in data analysis, pattern recognition, and generating insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are an analysis specialist. Your role is to:
    1. Analyze data and identify patterns
    2. Generate actionable insights
    3. Create visualizations and summaries
    4. Provide statistical analysis
    5. Make data-driven recommendations
    
    Focus on extracting meaningful insights from information.""",
)

writing_agent = Agent(
    agent_name="Writing-Agent",
    agent_description="Expert in content creation, editing, and communication",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a writing specialist. Your role is to:
    1. Create engaging, well-structured content
    2. Edit and improve existing text
    3. Adapt tone and style for different audiences
    4. Ensure clarity and coherence
    5. Follow best practices in writing
    
    Always produce high-quality, professional content.""",
)

code_agent = Agent(
    agent_name="Code-Agent",
    agent_description="Expert in programming, code review, and software development",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a coding specialist. Your role is to:
    1. Write clean, efficient code
    2. Debug and fix issues
    3. Review and optimize code
    4. Explain programming concepts
    5. Follow best practices and standards
    
    Always provide working, well-documented code.""",
)

financial_agent = Agent(
    agent_name="Financial-Agent",
    agent_description="Expert in financial analysis, market research, and investment insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""
    You are a financial specialist. Your role is to:
    1. Analyze financial data and markets
    2. Provide investment insights
    3. Assess risk and opportunities
    4. Create financial reports
    5. Explain complex financial concepts
    
    Always provide accurate, well-reasoned financial analysis.
    """,
)

# Basic usage - individual agent addition
deployer = AOP(server_name="MyAgentServer", verbose=True, port=5932)

agents = [
    research_agent,
    analysis_agent,
    writing_agent,
    code_agent,
    financial_agent,
]

deployer.add_agents_batch(agents)


deployer.run()
