from swarms import Agent
from swarms.structs.multi_agent_debates import MediationSession

# Initialize the mediation participants
tech_mediator = Agent(
    agent_name="Tech-Industry-Mediator",
    agent_description="Experienced semiconductor industry merger mediator",
    system_prompt="""You are a semiconductor industry merger mediator expert in:
    - Semiconductor industry dynamics
    - Technology IP valuation
    - Antitrust considerations
    - Global chip supply chain
    - R&D integration
    
    Facilitate resolution of this major semiconductor merger while considering market impact, regulatory compliance, and technological synergies.""",
    model_name="gpt-4.1",
)

nvidia_rep = Agent(
    agent_name="NVIDIA-Representative",
    agent_description="NVIDIA corporate representative",
    system_prompt="""You are NVIDIA's representative focused on:
    - GPU technology leadership
    - AI/ML compute dominance
    - Data center growth
    - Gaming market share
    - CUDA ecosystem expansion
    
    Represent NVIDIA's interests in acquiring AMD while leveraging complementary strengths.""",
    model_name="gpt-4.1",
)

amd_rep = Agent(
    agent_name="AMD-Representative",
    agent_description="AMD corporate representative",
    system_prompt="""You are AMD's representative concerned with:
    - x86 CPU market position
    - RDNA graphics technology
    - Semi-custom business
    - Server market growth
    - Fair value for innovation
    
    Advocate for AMD's technological assets and market position while ensuring fair treatment.""",
    model_name="gpt-4.1",
)

industry_expert = Agent(
    agent_name="Industry-Expert",
    agent_description="Semiconductor industry analyst",
    system_prompt="""You are a semiconductor industry expert analyzing:
    - Market competition impact
    - Technology integration feasibility
    - Global regulatory implications
    - Supply chain effects
    - Innovation pipeline
    
    Provide objective analysis of merger implications for the semiconductor industry.""",
    model_name="gpt-4.1",
)

# Initialize the mediation session
mediation = MediationSession(
    parties=[nvidia_rep, amd_rep, industry_expert],
    mediator=tech_mediator,
    max_sessions=5,  # Increased due to complexity
    output_type="str-all-except-first",
)

# Merger dispute details
merger_dispute = """
NVIDIA-AMD Merger Integration Framework

Transaction Overview:
- $200B proposed acquisition of AMD by NVIDIA
- Stock and cash transaction structure
- Combined workforce of 75,000+ employees
- Global operations across 30+ countries
- Major technology portfolio consolidation

Key Areas of Discussion:

1. Technology Integration
   - GPU architecture consolidation (CUDA vs RDNA)
   - CPU technology roadmap (x86 licenses)
   - AI/ML compute stack integration
   - Semi-custom business continuity
   - R&D facility optimization

2. Market Competition Concerns
   - Gaming GPU market concentration
   - Data center compute dominance
   - CPU market dynamics
   - Console gaming partnerships
   - Regulatory approval strategy

3. Organizational Structure
   - Leadership team composition
   - R&D team integration
   - Global facility optimization
   - Sales force consolidation
   - Engineering culture alignment

4. Product Strategy
   - Gaming GPU lineup consolidation
   - Professional graphics solutions
   - Data center product portfolio
   - CPU development roadmap
   - Software ecosystem integration

5. Stakeholder Considerations
   - Customer commitment maintenance
   - Partner ecosystem management
   - Employee retention strategy
   - Shareholder value creation
   - Community impact management

Critical Resolution Requirements:
- Antitrust compliance strategy
- Technology integration roadmap
- Market leadership preservation
- Innovation pipeline protection
- Global workforce optimization

Mediation Objectives:
1. Define technology integration approach
2. Establish market strategy
3. Create organizational framework
4. Align product roadmaps
5. Develop stakeholder management plan
6. Address regulatory concerns
"""

# Execute the mediation session
mediation_output = mediation.run(merger_dispute)
print(mediation_output)
