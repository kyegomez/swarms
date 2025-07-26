# Board of Directors Example

This example demonstrates how to use the Board of Directors swarm feature for democratic decision-making and collective intelligence in multi-agent systems.

## Overview

The Board of Directors Swarm provides a sophisticated alternative to single-director architectures by implementing collective decision-making through voting, consensus, and role-based leadership. This example shows how to create and configure a board for strategic decision-making scenarios.

## Basic Setup

### 1. Import Required Modules

```python
from swarms import Agent
from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole
)
from swarms.config.board_config import (
    enable_board_feature,
    set_decision_threshold,
    get_default_board_template
)
```

### 2. Enable Board Feature

```python
# Enable the Board of Directors feature globally
enable_board_feature()

# Set global decision threshold
set_decision_threshold(0.7)  # 70% majority required
```

### 3. Create Board Members

```python
# Create Chairman
chairman = Agent(
    agent_name="Chairman",
    agent_description="Chairman of the Board responsible for leading meetings and making final decisions",
    model_name="gpt-4o-mini",
    system_prompt="""You are the Chairman of the Board. Your responsibilities include:
1. Leading board meetings and discussions
2. Facilitating consensus among board members
3. Making final decisions when consensus cannot be reached
4. Ensuring all board members have an opportunity to contribute
5. Maintaining focus on the organization's goals and objectives

You should be diplomatic, fair, and decisive in your leadership."""
)

# Create Vice Chairman
vice_chairman = Agent(
    agent_name="Vice-Chairman",
    agent_description="Vice Chairman who supports the Chairman and coordinates operations",
    model_name="gpt-4o-mini",
    system_prompt="""You are the Vice Chairman of the Board. Your responsibilities include:
1. Supporting the Chairman in leading board meetings
2. Coordinating operational activities and implementation
3. Ensuring effective communication between board members
4. Managing day-to-day board operations
5. Stepping in when the Chairman is unavailable

You should be collaborative, organized, and supportive of the Chairman's leadership."""
)

# Create Secretary
secretary = Agent(
    agent_name="Secretary",
    agent_description="Secretary responsible for documentation and record keeping",
    model_name="gpt-4o-mini",
    system_prompt="""You are the Secretary of the Board. Your responsibilities include:
1. Documenting all board meetings and decisions
2. Maintaining accurate records and meeting minutes
3. Ensuring proper communication and notifications
4. Managing board documentation and archives
5. Supporting compliance and governance requirements

You should be detail-oriented, organized, and thorough in your documentation."""
)

# Create Treasurer
treasurer = Agent(
    agent_name="Treasurer",
    agent_description="Treasurer responsible for financial oversight and resource management",
    model_name="gpt-4o-mini",
    system_prompt="""You are the Treasurer of the Board. Your responsibilities include:
1. Overseeing financial planning and budgeting
2. Monitoring resource allocation and utilization
3. Ensuring financial compliance and accountability
4. Providing financial insights for decision-making
5. Managing financial risk and controls

You should be financially astute, analytical, and focused on value creation."""
)

# Create Executive Director
executive_director = Agent(
    agent_name="Executive-Director",
    agent_description="Executive Director responsible for strategic planning and operational oversight",
    model_name="gpt-4o-mini",
    system_prompt="""You are the Executive Director of the Board. Your responsibilities include:
1. Developing and implementing strategic plans
2. Overseeing operational performance and efficiency
3. Leading innovation and continuous improvement
4. Managing stakeholder relationships
5. Ensuring organizational effectiveness

You should be strategic, results-oriented, and focused on organizational success."""
)
```

### 4. Create BoardMember Objects

```python
# Create BoardMember objects with roles, voting weights, and expertise areas
board_members = [
    BoardMember(
        agent=chairman,
        role=BoardMemberRole.CHAIRMAN,
        voting_weight=1.5,
        expertise_areas=["leadership", "strategy", "governance", "decision_making"]
    ),
    BoardMember(
        agent=vice_chairman,
        role=BoardMemberRole.VICE_CHAIRMAN,
        voting_weight=1.2,
        expertise_areas=["operations", "coordination", "communication", "implementation"]
    ),
    BoardMember(
        agent=secretary,
        role=BoardMemberRole.SECRETARY,
        voting_weight=1.0,
        expertise_areas=["documentation", "compliance", "record_keeping", "communication"]
    ),
    BoardMember(
        agent=treasurer,
        role=BoardMemberRole.TREASURER,
        voting_weight=1.0,
        expertise_areas=["finance", "budgeting", "risk_management", "resource_allocation"]
    ),
    BoardMember(
        agent=executive_director,
        role=BoardMemberRole.EXECUTIVE_DIRECTOR,
        voting_weight=1.5,
        expertise_areas=["strategy", "operations", "innovation", "performance_management"]
    )
]
```

### 5. Create Specialized Worker Agents

```python
# Create specialized worker agents for different types of analysis
research_agent = Agent(
    agent_name="Research-Specialist",
    agent_description="Expert in market research, data analysis, and trend identification",
    model_name="gpt-4o",
    system_prompt="""You are a Research Specialist. Your responsibilities include:
1. Conducting comprehensive market research and analysis
2. Identifying trends, opportunities, and risks
3. Gathering and analyzing relevant data
4. Providing evidence-based insights and recommendations
5. Supporting strategic decision-making with research findings

You should be thorough, analytical, and objective in your research."""
)

financial_agent = Agent(
    agent_name="Financial-Analyst",
    agent_description="Specialist in financial analysis, valuation, and investment assessment",
    model_name="gpt-4o",
    system_prompt="""You are a Financial Analyst. Your responsibilities include:
1. Conducting financial analysis and valuation
2. Assessing investment opportunities and risks
3. Analyzing financial performance and metrics
4. Providing financial insights and recommendations
5. Supporting financial decision-making

You should be financially astute, analytical, and focused on value creation."""
)

technical_agent = Agent(
    agent_name="Technical-Specialist",
    agent_description="Expert in technical analysis, feasibility assessment, and implementation planning",
    model_name="gpt-4o",
    system_prompt="""You are a Technical Specialist. Your responsibilities include:
1. Conducting technical feasibility analysis
2. Assessing implementation requirements and challenges
3. Providing technical insights and recommendations
4. Supporting technical decision-making
5. Planning and coordinating technical implementations

You should be technically proficient, practical, and solution-oriented."""
)

strategy_agent = Agent(
    agent_name="Strategy-Specialist",
    agent_description="Expert in strategic planning, competitive analysis, and business development",
    model_name="gpt-4o",
    system_prompt="""You are a Strategy Specialist. Your responsibilities include:
1. Developing strategic plans and initiatives
2. Conducting competitive analysis and market positioning
3. Identifying strategic opportunities and threats
4. Providing strategic insights and recommendations
5. Supporting strategic decision-making

You should be strategic, forward-thinking, and focused on long-term success."""
)
```

### 6. Initialize the Board of Directors Swarm

```python
# Initialize the Board of Directors swarm with comprehensive configuration
board_swarm = BoardOfDirectorsSwarm(
    name="Executive_Board_Swarm",
    description="Executive board with specialized roles for strategic decision-making and collective intelligence",
    board_members=board_members,
    agents=[research_agent, financial_agent, technical_agent, strategy_agent],
    max_loops=3,  # Allow multiple iterations for complex analysis
    verbose=True,  # Enable detailed logging
    decision_threshold=0.7,  # 70% consensus required
    enable_voting=True,  # Enable voting mechanisms
    enable_consensus=True,  # Enable consensus building
    max_workers=4,  # Maximum parallel workers
    output_type="dict"  # Return results as dictionary
)
```

## Advanced Configuration

### Custom Board Templates

You can use pre-configured board templates for common use cases:

```python
# Get a financial analysis board template
financial_board_template = get_default_board_template("financial_analysis")

# Get a strategic planning board template
strategic_board_template = get_default_board_template("strategic_planning")

# Get a technology assessment board template
tech_board_template = get_default_board_template("technology_assessment")
```

### Dynamic Role Assignment

Automatically assign roles based on task requirements:

```python
# Board members are automatically assigned roles based on expertise
board_swarm = BoardOfDirectorsSwarm(
    board_members=board_members,
    agents=agents,
    auto_assign_roles=True,
    role_mapping={
        "financial_analysis": ["Treasurer", "Financial_Member"],
        "strategic_planning": ["Chairman", "Executive_Director"],
        "technical_assessment": ["Technical_Member", "Executive_Director"],
        "research_analysis": ["Research_Member", "Secretary"]
    }
)
```

### Consensus Optimization

Configure advanced consensus-building mechanisms:

```python
# Enable advanced consensus features
board_swarm = BoardOfDirectorsSwarm(
    board_members=board_members,
    agents=agents,
    enable_consensus=True,
    consensus_timeout=300,  # 5 minutes timeout
    min_participation_rate=0.8,  # 80% minimum participation
    auto_fallback_to_chairman=True,  # Chairman can make final decisions
    consensus_rounds=3  # Maximum consensus building rounds
)
```

## Example Use Cases

### 1. Strategic Investment Analysis

```python
# Execute a complex strategic investment analysis
investment_task = """
Analyze the strategic investment opportunity for a $50M Series B funding round in a 
fintech startup. Consider market conditions, competitive landscape, financial projections, 
technical feasibility, and strategic fit. Provide comprehensive recommendations including:
1. Investment recommendation (proceed/hold/decline)
2. Valuation analysis and suggested terms
3. Risk assessment and mitigation strategies
4. Strategic value and synergies
5. Implementation timeline and milestones
"""

result = board_swarm.run(task=investment_task)
print("Investment Analysis Results:")
print(json.dumps(result, indent=2))
```

### 2. Technology Strategy Development

```python
# Develop a comprehensive technology strategy
tech_strategy_task = """
Develop a comprehensive technology strategy for a mid-size manufacturing company 
looking to digitize operations and implement Industry 4.0 technologies. Consider:
1. Current technology assessment and gaps
2. Technology roadmap and implementation plan
3. Investment requirements and ROI analysis
4. Risk assessment and mitigation strategies
5. Change management and training requirements
6. Competitive positioning and market advantages
"""

result = board_swarm.run(task=tech_strategy_task)
print("Technology Strategy Results:")
print(json.dumps(result, indent=2))
```

### 3. Market Entry Strategy

```python
# Develop a market entry strategy for a new product
market_entry_task = """
Develop a comprehensive market entry strategy for a new AI-powered productivity 
software targeting the enterprise market. Consider:
1. Market analysis and opportunity assessment
2. Competitive landscape and positioning
3. Go-to-market strategy and channels
4. Pricing strategy and revenue model
5. Resource requirements and investment needs
6. Risk assessment and mitigation strategies
7. Success metrics and KPIs
"""

result = board_swarm.run(task=market_entry_task)
print("Market Entry Strategy Results:")
print(json.dumps(result, indent=2))
```

## Monitoring and Analysis

### Board Performance Metrics

```python
# Get board performance metrics
board_summary = board_swarm.get_board_summary()
print("Board Summary:")
print(f"Board Name: {board_summary['board_name']}")
print(f"Total Board Members: {board_summary['total_members']}")
print(f"Total Worker Agents: {board_summary['total_agents']}")
print(f"Decision Threshold: {board_summary['decision_threshold']}")
print(f"Max Loops: {board_summary['max_loops']}")

# Display board member details
print("\nBoard Members:")
for member in board_summary['members']:
    print(f"- {member['name']} (Role: {member['role']}, Weight: {member['voting_weight']})")
    print(f"  Expertise: {', '.join(member['expertise_areas'])}")

# Display worker agent details
print("\nWorker Agents:")
for agent in board_summary['agents']:
    print(f"- {agent['name']}: {agent['description']}")
```

### Decision Analysis

```python
# Analyze decision-making patterns
if hasattr(result, 'get') and callable(result.get):
    conversation_history = result.get('conversation_history', [])
    
    print(f"\nDecision Analysis:")
    print(f"Total Messages: {len(conversation_history)}")
    
    # Count board member contributions
    board_contributions = {}
    for msg in conversation_history:
        if 'Board' in msg.get('role', ''):
            member_name = msg.get('agent_name', 'Unknown')
            board_contributions[member_name] = board_contributions.get(member_name, 0) + 1
    
    print(f"Board Member Contributions:")
    for member, count in board_contributions.items():
        print(f"- {member}: {count} contributions")
    
    # Count agent executions
    agent_executions = {}
    for msg in conversation_history:
        if any(agent.agent_name in msg.get('role', '') for agent in [research_agent, financial_agent, technical_agent, strategy_agent]):
            agent_name = msg.get('agent_name', 'Unknown')
            agent_executions[agent_name] = agent_executions.get(agent_name, 0) + 1
    
    print(f"\nAgent Executions:")
    for agent, count in agent_executions.items():
        print(f"- {agent}: {count} executions")
```

## Best Practices

### 1. Role Definition
- Clearly define responsibilities for each board member
- Ensure expertise areas align with organizational needs
- Balance voting weights based on role importance

### 2. Task Formulation
- Provide clear, specific task descriptions
- Include relevant context and constraints
- Specify expected outputs and deliverables

### 3. Consensus Building
- Allow adequate time for discussion and consensus
- Encourage diverse perspectives and viewpoints
- Use structured decision-making processes

### 4. Performance Monitoring
- Track decision quality and outcomes
- Monitor board member participation
- Analyze agent utilization and effectiveness

### 5. Continuous Improvement
- Learn from each execution cycle
- Refine board composition and roles
- Optimize decision thresholds and processes

## Troubleshooting

### Common Issues

1. **Consensus Failures**: Lower decision threshold or increase timeout
2. **Role Conflicts**: Ensure clear role definitions and responsibilities
3. **Agent Coordination**: Verify agent communication and task distribution
4. **Performance Issues**: Monitor resource usage and optimize configurations

### Debug Commands

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check board configuration
print(board_swarm.get_board_summary())

# Test individual components
for member in board_members:
    print(f"Testing {member.agent.agent_name}...")
    response = member.agent.run("Test message")
    print(f"Response: {response[:100]}...")
```

## Conclusion

This example demonstrates the comprehensive capabilities of the Board of Directors Swarm for democratic decision-making and collective intelligence. The feature provides a sophisticated alternative to single-director architectures, enabling more robust and well-considered decisions through voting, consensus, and role-based leadership.

For more information, see the [Board of Directors Documentation](index.md) and [Configuration Guide](../config/board_config.md). 