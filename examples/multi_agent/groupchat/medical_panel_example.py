"""Medical Panel Discussion Example.

A panel of medical specialists collaborates on a complex multi-system case
using the dynamic GroupChat. Each specialist independently decides when to
speak instead of following a fixed turn order.
"""

from swarms import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL


def create_medical_panel():
    """Create a panel of medical specialists for discussion."""

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
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

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
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

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
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

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
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

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
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

    return [
        cardiologist,
        oncologist,
        neurologist,
        endocrinologist,
        infectious_disease,
    ]


def example_medical_panel():
    """Example with the dynamic medical panel."""
    print("=== MEDICAL PANEL DISCUSSION ===\n")

    agents = create_medical_panel()

    group_chat = GroupChat(
        name="Medical Panel Discussion",
        description="A collaborative panel of medical specialists discussing complex cases",
        agents=agents,
        max_loops=25,
        threshold=0.5,
        idle_timeout=10.0,
    )

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
    example_medical_panel()
