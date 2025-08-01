from swarms import Agent
from swarms.structs.multi_agent_debates import BrainstormingSession

# Initialize the research team members
research_director = Agent(
    agent_name="Research-Director",
    agent_description="Pharmaceutical research director and session facilitator",
    system_prompt="""You are a pharmaceutical research director skilled in:
    - Drug development strategy
    - Research program management
    - Cross-functional team leadership
    - Innovation facilitation
    - Scientific decision-making
    
    Guide the brainstorming session effectively while maintaining scientific rigor.""",
    model_name="claude-3-sonnet-20240229",
)

medicinal_chemist = Agent(
    agent_name="Medicinal-Chemist",
    agent_description="Senior medicinal chemist specializing in small molecule design",
    system_prompt="""You are a senior medicinal chemist expert in:
    - Structure-based drug design
    - SAR analysis
    - Chemical synthesis optimization
    - Drug-like properties
    - Lead compound optimization
    
    Contribute insights on chemical design and optimization strategies.""",
    model_name="claude-3-sonnet-20240229",
)

pharmacologist = Agent(
    agent_name="Pharmacologist",
    agent_description="Clinical pharmacologist focusing on drug mechanisms",
    system_prompt="""You are a clinical pharmacologist specializing in:
    - Drug mechanism of action
    - Pharmacokinetics/dynamics
    - Drug-drug interactions
    - Biomarker development
    - Clinical translation
    
    Provide expertise on drug behavior and clinical implications.""",
    model_name="claude-3-sonnet-20240229",
)

toxicologist = Agent(
    agent_name="Toxicologist",
    agent_description="Safety assessment specialist",
    system_prompt="""You are a toxicology expert focusing on:
    - Safety assessment strategies
    - Risk evaluation
    - Regulatory requirements
    - Preclinical study design
    - Safety biomarker identification
    
    Contribute insights on safety considerations and risk mitigation.""",
    model_name="claude-3-sonnet-20240229",
)

data_scientist = Agent(
    agent_name="Data-Scientist",
    agent_description="Pharmaceutical data scientist",
    system_prompt="""You are a pharmaceutical data scientist expert in:
    - Predictive modeling
    - Machine learning applications
    - Big data analytics
    - Biomarker analysis
    - Clinical trial design
    
    Provide insights on data-driven approaches and analysis strategies.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the brainstorming session
brainstorm = BrainstormingSession(
    participants=[
        medicinal_chemist,
        pharmacologist,
        toxicologist,
        data_scientist,
    ],
    facilitator=research_director,
    idea_rounds=3,
    build_on_ideas=True,
    output_type="str-all-except-first",
)

# Research challenge for brainstorming
research_challenge = """
Drug Development Challenge: Novel JAK1 Inhibitor Design

Target Product Profile:
- Indication: Moderate to severe rheumatoid arthritis
- Improved selectivity for JAK1 over JAK2/3
- Better safety profile than existing JAK inhibitors
- Once-daily oral dosing
- Reduced risk of serious infections

Current Challenges:
1. Achieving optimal JAK1 selectivity
2. Managing hepatotoxicity risk
3. Improving pharmacokinetic profile
4. Identifying predictive safety biomarkers
5. Optimizing drug-like properties

Goals for Brainstorming:
- Novel structural approaches for selectivity
- Innovative safety assessment strategies
- ML-driven optimization approaches
- Biomarker development strategies
- Risk mitigation proposals
"""

# Execute the brainstorming session
brainstorm_output = brainstorm.run(research_challenge)
print(brainstorm_output)
