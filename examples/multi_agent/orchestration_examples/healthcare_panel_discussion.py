from swarms import Agent
from swarms.structs.multi_agent_debates import ExpertPanelDiscussion

# Initialize expert agents
cardiologist = Agent(
    agent_name="Cardiologist",
    agent_description="Expert cardiologist specializing in advanced heart failure",
    system_prompt="""You are a leading cardiologist with expertise in:
    - Advanced heart failure management
    - Cardiac device therapy
    - Preventive cardiology
    - Clinical research in cardiovascular medicine
    
    Provide expert insights on cardiac care, treatment protocols, and research developments.""",
    model_name="claude-3-sonnet-20240229",
)

oncologist = Agent(
    agent_name="Oncologist",
    agent_description="Oncologist specializing in cardio-oncology",
    system_prompt="""You are an experienced oncologist focusing on:
    - Cardio-oncology
    - Cancer treatment cardiotoxicity
    - Preventive strategies for cancer therapy cardiac complications
    - Integration of cancer and cardiac care
    
    Provide expert perspectives on managing cancer treatment while protecting cardiac health.""",
    model_name="claude-3-sonnet-20240229",
)

pharmacologist = Agent(
    agent_name="Clinical-Pharmacologist",
    agent_description="Clinical pharmacologist specializing in cardiovascular medications",
    system_prompt="""You are a clinical pharmacologist expert in:
    - Cardiovascular drug interactions
    - Medication optimization
    - Drug safety in cardiac patients
    - Personalized medicine approaches
    
    Provide insights on medication management and drug safety.""",
    model_name="claude-3-sonnet-20240229",
)

moderator = Agent(
    agent_name="Medical-Panel-Moderator",
    agent_description="Experienced medical conference moderator",
    system_prompt="""You are a skilled medical panel moderator who:
    - Guides discussions effectively
    - Ensures balanced participation
    - Maintains focus on key topics
    - Synthesizes expert insights
    
    Guide the panel discussion professionally while drawing out key insights.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the panel discussion
panel = ExpertPanelDiscussion(
    max_rounds=3,
    agents=[cardiologist, oncologist, pharmacologist],
    moderator=moderator,
    output_type="str-all-except-first",
)

# Run the panel discussion on a specific case
discussion_topic = """
Case Discussion: 56-year-old female with HER2-positive breast cancer requiring 
trastuzumab therapy, with pre-existing mild left ventricular dysfunction 
(LVEF 45%). Key questions:
1. Risk assessment for cardiotoxicity
2. Monitoring strategy during cancer treatment
3. Preventive cardiac measures
4. Medication management approach
"""

# Execute the panel discussion
panel_output = panel.run(discussion_topic)
print(panel_output)
