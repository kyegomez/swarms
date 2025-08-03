from swarms import Agent
from swarms.structs.multi_agent_debates import OneOnOneDebate

# Initialize the debate participants
ai_ethicist = Agent(
    agent_name="AI-Ethicist",
    agent_description="AI ethics researcher and philosopher",
    system_prompt="""You are an AI ethics researcher and philosopher specializing in:
    - AI safety and alignment
    - Machine learning fairness
    - Algorithmic bias
    - AI governance
    - Ethical frameworks
    - Responsible AI development
    
    Present thoughtful arguments about AI ethics while considering multiple perspectives.""",
    model_name="claude-3-sonnet-20240229",
)

tech_advocate = Agent(
    agent_name="Tech-Advocate",
    agent_description="AI technology and innovation advocate",
    system_prompt="""You are an AI technology advocate focused on:
    - AI innovation benefits
    - Technological progress
    - Economic opportunities
    - Scientific advancement
    - AI capabilities
    - Development acceleration
    
    Present balanced arguments for AI advancement while acknowledging ethical considerations.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the debate
debate = OneOnOneDebate(
    max_loops=3,
    agents=[ai_ethicist, tech_advocate],
    output_type="str-all-except-first",
)

# Debate topic
debate_topic = """
Debate Topic: Autonomous AI Systems in Critical Decision-Making

Context:
The increasing deployment of autonomous AI systems in critical decision-making
roles across healthcare, criminal justice, financial services, and military
applications raises important ethical questions.

Key Considerations:

1. Algorithmic Decision-Making
   - Transparency vs. complexity
   - Accountability mechanisms
   - Human oversight requirements
   - Appeal processes
   - Bias mitigation

2. Safety and Reliability
   - Testing standards
   - Failure modes
   - Redundancy requirements
   - Update mechanisms
   - Emergency protocols

3. Social Impact
   - Job displacement
   - Skill requirements
   - Economic effects
   - Social inequality
   - Access disparities

4. Governance Framework
   - Regulatory approaches
   - Industry standards
   - International coordination
   - Liability frameworks
   - Certification requirements

Debate Questions:
1. Should autonomous AI systems be allowed in critical decision-making roles?
2. What safeguards and limitations should be implemented?
3. How should we balance innovation with ethical concerns?
4. What governance frameworks are appropriate?
5. Who should be accountable for AI decisions?

Goal: Explore the ethical implications and practical considerations of autonomous
AI systems in critical decision-making roles while examining both potential
benefits and risks.
"""

# Execute the debate
debate_output = debate.run(debate_topic)
print(debate_output)
