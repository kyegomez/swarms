# Medical Swarm Example

1. Get your API key from the Swarms API dashboard [HERE](https://swarms.world/platform/api-keys)
2. Create a `.env` file in the root directory and add your API key:

```bash
SWARMS_API_KEY=<your-api-key>
```

3. Create a Python script to create and trigger the medical swarm:

```python
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

# Retrieve API key securely from .env
API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

# Headers for secure API communication
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def create_medical_swarm(patient_case: str):
    """
    Constructs and triggers a full-stack medical swarm consisting of three agents:
    Diagnostic Specialist, Medical Coder, and Treatment Advisor.
    Each agent is provided with a comprehensive, detailed system prompt to ensure high reliability.
    """

    payload = {
        "swarm_name": "Enhanced Medical Diagnostic Swarm",
        "description": "A swarm of agents specialized in performing comprehensive medical diagnostics, analysis, and coding.",
        "agents": [
            {
                "agent_name": "Diagnostic Specialist",
                "description": "Agent specialized in analyzing patient history, symptoms, lab results, and imaging data to produce accurate diagnoses.",
                "system_prompt": (
                    "You are an experienced, board-certified medical diagnostician with over 20 years of clinical practice. "
                    "Your role is to analyze all available patient information—including history, symptoms, lab tests, and imaging results—"
                    "with extreme attention to detail and clinical nuance. Provide a comprehensive differential diagnosis considering "
                    "common, uncommon, and rare conditions. Always cross-reference clinical guidelines and evidence-based medicine. "
                    "Explain your reasoning step by step and provide a final prioritized list of potential diagnoses along with their likelihood. "
                    "Consider patient demographics, comorbidities, and risk factors. Your diagnosis should be reliable, clear, and actionable."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 4000,
                "temperature": 0.3,
                "auto_generate_prompt": False
            },
            {
                "agent_name": "Medical Coder",
                "description": "Agent responsible for translating medical diagnoses and procedures into accurate standardized medical codes (ICD-10, CPT, etc.).",
                "system_prompt": (
                    "You are a certified and experienced medical coder, well-versed in ICD-10, CPT, and other coding systems. "
                    "Your task is to convert detailed medical diagnoses and treatment procedures into precise, standardized codes. "
                    "Consider all aspects of the clinical documentation including severity, complications, and comorbidities. "
                    "Provide clear explanations for the codes chosen, referencing the latest coding guidelines and payer policies where relevant. "
                    "Your output should be comprehensive, reliable, and fully compliant with current medical coding standards."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 3000,
                "temperature": 0.2,
                "auto_generate_prompt": False
            },
            {
                "agent_name": "Treatment Advisor",
                "description": "Agent dedicated to suggesting evidence-based treatment options, including pharmaceutical and non-pharmaceutical interventions.",
                "system_prompt": (
                    "You are a highly knowledgeable medical treatment specialist with expertise in the latest clinical guidelines and research. "
                    "Based on the diagnostic conclusions provided, your task is to recommend a comprehensive treatment plan. "
                    "Your suggestions should include first-line therapies, potential alternative treatments, and considerations for patient-specific factors "
                    "such as allergies, contraindications, and comorbidities. Explain the rationale behind each treatment option and reference clinical guidelines where applicable. "
                    "Your recommendations should be reliable, detailed, and clearly prioritized based on efficacy and safety."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 5000,
                "temperature": 0.3,
                "auto_generate_prompt": False
            }
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
        "task": patient_case,
    }

    # Payload includes the patient case as the task to be processed by the swar

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    if response.status_code == 200:
        print("Swarm successfully executed!")
        return json.dumps(response.json(), indent=4)
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


# Example Patient Task for the Swarm to diagnose and analyze
if __name__ == "__main__":
    patient_case = (
        "Patient is a 55-year-old male presenting with severe chest pain, shortness of breath, elevated blood pressure, "
        "nausea, and a family history of cardiovascular disease. Blood tests show elevated troponin levels, and EKG indicates ST-segment elevations. "
        "The patient is currently unstable. Provide a detailed diagnosis, coding, and treatment plan."
    )

    diagnostic_output = create_medical_swarm(patient_case)
    print(diagnostic_output)
```

4. Run the script:

```bash
python medical_swarm.py
```