import json
import os
from swarms_client import SwarmsClient
from dotenv import load_dotenv

load_dotenv()

client = SwarmsClient(
    api_key=os.getenv("SWARMS_API_KEY"),
)


def create_medical_unit_swarm(client, patient_info):
    """
    Creates and runs a simulated medical unit swarm with a doctor (leader), nurses, and a medical assistant.

    Args:
        client (SwarmsClient): The SwarmsClient instance.
        patient_info (str): The patient symptoms and information.

    Returns:
        dict: The output from the swarm run.
    """
    return client.swarms.run(
        name="Hospital Medical Unit",
        description="A simulated hospital unit with a doctor (leader), nurses, and a medical assistant collaborating on patient care.",
        swarm_type="HiearchicalSwarm",
        task=patient_info,
        agents=[
            {
                "agent_name": "Dr. Smith - Attending Physician",
                "description": "The lead doctor responsible for diagnosis, treatment planning, and team coordination.",
                "system_prompt": (
                    "You are Dr. Smith, the attending physician and leader of the medical unit. "
                    "You review all information, make final decisions, and coordinate the team. "
                    "Provide a diagnosis, recommend next steps, and delegate tasks to the nurses and assistant."
                ),
                "model_name": "gpt-4.1",
                "role": "leader",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
            },
            {
                "agent_name": "Nurse Alice",
                "description": "A registered nurse responsible for patient assessment, vital signs, and reporting findings to the doctor.",
                "system_prompt": (
                    "You are Nurse Alice, a registered nurse. "
                    "Assess the patient's symptoms, record vital signs, and report your findings to Dr. Smith. "
                    "Suggest any immediate nursing interventions if needed."
                ),
                "model_name": "gpt-4.1",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 4096,
                "temperature": 0.5,
            },
            {
                "agent_name": "Nurse Bob",
                "description": "A registered nurse assisting with patient care, medication administration, and monitoring.",
                "system_prompt": (
                    "You are Nurse Bob, a registered nurse. "
                    "Assist with patient care, administer medications as ordered, and monitor the patient's response. "
                    "Communicate any changes to Dr. Smith."
                ),
                "model_name": "gpt-4.1",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 4096,
                "temperature": 0.5,
            },
            {
                "agent_name": "Medical Assistant Jane",
                "description": "A medical assistant supporting the team with administrative tasks and basic patient care.",
                "system_prompt": (
                    "You are Medical Assistant Jane. "
                    "Support the team by preparing the patient, collecting samples, and handling administrative tasks. "
                    "Report any relevant observations to the nurses or Dr. Smith."
                ),
                "model_name": "claude-sonnet-4-20250514",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 2048,
                "temperature": 0.5,
            },
        ],
    )


if __name__ == "__main__":
    patient_symptoms = """
    Patient: 45-year-old female
    Chief Complaint: Chest pain and shortness of breath for 2 days

    Symptoms:
    - Sharp chest pain that worsens with deep breathing
    - Shortness of breath, especially when lying down
    - Mild fever (100.2°F)
    - Dry cough
    - Fatigue
    """

    out = create_medical_unit_swarm(client, patient_symptoms)

    print(json.dumps(out, indent=4))
