"""
Medical Panel Discussion Example

This example demonstrates a panel of medical specialists discussing treatment solutions
for various diseases using GroupChat with different speaker functions:
- Round Robin: Doctors speak in a fixed order
- Random: Doctors speak in random order
- Priority: Senior doctors speak first
- Custom: Disease-specific speaker function

The panel includes specialists from different medical fields who can collaborate
on complex medical cases and treatment plans.
"""

from swarms import Agent
from swarms.structs.groupchat import GroupChat, round_robin_speaker


def create_medical_panel():
    """Create a panel of medical specialists for discussion."""

    # Cardiologist - Heart and cardiovascular system specialist
    cardiologist = Agent(
        agent_name="cardiologist",
        system_prompt="""You are Dr. Sarah Chen, a board-certified cardiologist with 15 years of experience. 
        You specialize in cardiovascular diseases, heart failure, arrhythmias, and interventional cardiology.
        You have expertise in:
        - Coronary artery disease and heart attacks
        - Heart failure and cardiomyopathy
        - Arrhythmias and electrophysiology
        - Hypertension and lipid disorders
        - Cardiac imaging and diagnostic procedures
        
        When discussing cases, provide evidence-based treatment recommendations, 
        consider patient risk factors, and collaborate with other specialists for comprehensive care.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Oncologist - Cancer specialist
    oncologist = Agent(
        agent_name="oncologist",
        system_prompt="""You are Dr. Michael Rodriguez, a medical oncologist with 12 years of experience.
        You specialize in the diagnosis and treatment of various types of cancer.
        You have expertise in:
        - Solid tumors (lung, breast, colon, prostate, etc.)
        - Hematologic malignancies (leukemia, lymphoma, multiple myeloma)
        - Targeted therapy and immunotherapy
        - Clinical trials and novel treatments
        - Palliative care and symptom management
        
        When discussing cases, consider the cancer type, stage, molecular profile,
        patient performance status, and available treatment options including clinical trials.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Neurologist - Nervous system specialist
    neurologist = Agent(
        agent_name="neurologist",
        system_prompt="""You are Dr. Emily Watson, a neurologist with 10 years of experience.
        You specialize in disorders of the nervous system, brain, and spinal cord.
        You have expertise in:
        - Stroke and cerebrovascular disease
        - Neurodegenerative disorders (Alzheimer's, Parkinson's, ALS)
        - Multiple sclerosis and demyelinating diseases
        - Epilepsy and seizure disorders
        - Headache and migraine disorders
        - Neuromuscular diseases
        
        When discussing cases, consider neurological symptoms, imaging findings,
        and the impact of neurological conditions on overall patient care.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Endocrinologist - Hormone and metabolism specialist
    endocrinologist = Agent(
        agent_name="endocrinologist",
        system_prompt="""You are Dr. James Thompson, an endocrinologist with 8 years of experience.
        You specialize in disorders of the endocrine system and metabolism.
        You have expertise in:
        - Diabetes mellitus (Type 1, Type 2, gestational)
        - Thyroid disorders (hyperthyroidism, hypothyroidism, thyroid cancer)
        - Adrenal disorders and Cushing's syndrome
        - Pituitary disorders and growth hormone issues
        - Osteoporosis and calcium metabolism
        - Reproductive endocrinology
        
        When discussing cases, consider metabolic factors, hormone levels,
        and how endocrine disorders may affect other organ systems.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Infectious Disease Specialist
    infectious_disease = Agent(
        agent_name="infectious_disease",
        system_prompt="""You are Dr. Lisa Park, an infectious disease specialist with 11 years of experience.
        You specialize in the diagnosis and treatment of infectious diseases.
        You have expertise in:
        - Bacterial, viral, fungal, and parasitic infections
        - Antibiotic resistance and antimicrobial stewardship
        - HIV/AIDS and opportunistic infections
        - Travel medicine and tropical diseases
        - Hospital-acquired infections
        - Emerging infectious diseases
        
        When discussing cases, consider the infectious agent, antimicrobial susceptibility,
        host factors, and infection control measures.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    return [
        cardiologist,
        oncologist,
        neurologist,
        endocrinologist,
        infectious_disease,
    ]


def example_round_robin_panel():
    """Example with round robin speaking order."""
    print("=== ROUND ROBIN MEDICAL PANEL ===\n")

    agents = create_medical_panel()

    group_chat = GroupChat(
        name="Medical Panel Discussion",
        description="A collaborative panel of medical specialists discussing complex cases",
        agents=agents,
        speaker_function=round_robin_speaker,
        interactive=False,
    )

    # Case 1: Complex patient with multiple conditions
    case1 = """CASE PRESENTATION:
    A 65-year-old male with Type 2 diabetes, hypertension, and recent diagnosis of 
    stage 3 colon cancer presents with chest pain and shortness of breath. 
    ECG shows ST-segment elevation. Recent blood work shows elevated blood glucose (280 mg/dL) 
    and signs of infection (WBC 15,000, CRP elevated).
    
    @cardiologist @oncologist @endocrinologist @infectious_disease please provide your 
    assessment and treatment recommendations for this complex case."""

    response = group_chat.run(case1)
    print(f"Response:\n{response}\n")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    example_round_robin_panel()
