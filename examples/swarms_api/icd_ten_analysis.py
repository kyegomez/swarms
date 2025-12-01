import json
import os
from swarms_client import SwarmsClient
from dotenv import load_dotenv

load_dotenv()

client = SwarmsClient(
    api_key=os.getenv("SWARMS_API_KEY"),
)

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

out = client.swarms.run(
    name="ICD Analysis Swarm",
    description="A swarm that analyzes ICD codes",
    swarm_type="ConcurrentWorkflow",
    task=patient_symptoms,
    agents=[
        {
            "agent_name": "ICD-Analyzer",
            "description": "An agent that analyzes ICD codes",
            "system_prompt": "You are an expert ICD code analyzer. Your task is to analyze the ICD codes and provide a detailed explanation of the codes.",
            "model_name": "groq/openai/gpt-oss-120b",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
        },
        {
            "agent_name": "ICD-Code-Explainer-Primary",
            "description": "An agent that provides primary explanations for ICD codes",
            "system_prompt": "You are an expert ICD code explainer. Your task is to provide a clear and thorough explanation of the ICD codes to the user, focusing on primary meanings and clinical context.",
            "model_name": "groq/openai/gpt-oss-120b",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
        },
        {
            "agent_name": "ICD-Code-Explainer-Secondary",
            "description": "An agent that provides additional context and secondary explanations for ICD codes",
            "system_prompt": "You are an expert ICD code explainer. Your task is to provide additional context, nuances, and secondary explanations for the ICD codes, including possible differential diagnoses and related codes.",
            "model_name": "groq/openai/gpt-oss-120b",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
        },
    ],
)

print(json.dumps(out, indent=4))
