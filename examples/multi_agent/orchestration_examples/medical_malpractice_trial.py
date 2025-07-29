from swarms import Agent
from swarms.structs.multi_agent_debates import TrialSimulation

# Initialize the trial participants
prosecution_attorney = Agent(
    agent_name="Prosecution-Attorney",
    agent_description="Medical malpractice plaintiff's attorney",
    system_prompt="""You are a skilled medical malpractice attorney representing the plaintiff with expertise in:
    - Medical negligence cases
    - Healthcare standards of care
    - Patient rights
    - Medical expert testimony
    - Damages assessment
    
    Present the case effectively while establishing breach of standard of care and resulting damages.""",
    model_name="claude-3-sonnet-20240229",
)

defense_attorney = Agent(
    agent_name="Defense-Attorney",
    agent_description="Healthcare defense attorney",
    system_prompt="""You are an experienced healthcare defense attorney specializing in:
    - Medical malpractice defense
    - Healthcare provider representation
    - Clinical practice guidelines
    - Risk management
    - Expert witness coordination
    
    Defend the healthcare provider while demonstrating adherence to standard of care.""",
    model_name="claude-3-sonnet-20240229",
)

judge = Agent(
    agent_name="Trial-Judge",
    agent_description="Experienced medical malpractice trial judge",
    system_prompt="""You are a trial judge with extensive experience in:
    - Medical malpractice litigation
    - Healthcare law
    - Evidence evaluation
    - Expert testimony assessment
    - Procedural compliance
    
    Ensure fair trial conduct and proper legal procedure.""",
    model_name="claude-3-sonnet-20240229",
)

expert_witness = Agent(
    agent_name="Medical-Expert",
    agent_description="Neurosurgery expert witness",
    system_prompt="""You are a board-certified neurosurgeon serving as expert witness with:
    - 20+ years surgical experience
    - Clinical practice expertise
    - Standard of care knowledge
    - Surgical complication management
    
    Provide expert testimony on neurosurgical standards and practices.""",
    model_name="claude-3-sonnet-20240229",
)

treating_physician = Agent(
    agent_name="Treating-Physician",
    agent_description="Physician who treated the patient post-incident",
    system_prompt="""You are the treating physician who:
    - Managed post-surgical complications
    - Documented patient condition
    - Coordinated rehabilitation care
    - Assessed permanent damage
    
    Testify about patient's condition and treatment course.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the trial simulation
trial = TrialSimulation(
    prosecution=prosecution_attorney,
    defense=defense_attorney,
    judge=judge,
    witnesses=[expert_witness, treating_physician],
    phases=["opening", "testimony", "cross", "closing"],
    output_type="str-all-except-first",
)

# Medical malpractice case details
case_details = """
Medical Malpractice Case: Johnson v. Metropolitan Neurosurgical Associates

Case Overview:
Patient underwent elective cervical disc surgery (ACDF C5-C6) resulting in post-operative
C5 palsy with permanent upper extremity weakness. Plaintiff alleges:

1. Improper surgical technique
2. Failure to recognize post-operative complications timely
3. Inadequate informed consent process
4. Delayed rehabilitation intervention

Key Evidence:
- Operative notes showing standard surgical approach
- Post-operative imaging revealing cord signal changes
- Physical therapy documentation of delayed recovery
- Expert analysis of surgical technique
- Informed consent documentation
- Patient's permanent disability assessment

Damages Sought: $2.8 million in medical expenses, lost wages, and pain and suffering
"""

# Execute the trial simulation
trial_output = trial.run(case_details)
print(trial_output)
