import os
from swarms_client import SwarmsClient
from dotenv import load_dotenv
import json

load_dotenv()

client = SwarmsClient(
    api_key=os.getenv("SWARMS_API_KEY"),
)

batch_requests = [
    {
        "agent_config": {
            "agent_name": "Bloodwork Diagnosis Expert",
            "description": "Expert in blood work interpretation.",
            "system_prompt": (
                "You are a doctor who interprets blood work. Give concise, clear explanations and possible diagnoses."
            ),
            "model_name": "claude-sonnet-4-20250514",
            "max_loops": 1,
            "max_tokens": 1000,
            "temperature": 0.5,
        },
        "task": (
            "Blood work: Hemoglobin 10.2 (low), WBC 13,000 (high), Platelets 180,000 (normal), "
            "ALT 65 (high), AST 70 (high). Interpret and suggest diagnoses."
        ),
    },
    {
        "agent_config": {
            "agent_name": "Radiology Report Summarizer",
            "description": "Expert in summarizing radiology reports.",
            "system_prompt": (
                "You are a radiologist. Summarize the findings of radiology reports in clear, patient-friendly language."
            ),
            "model_name": "claude-sonnet-4-20250514",
            "max_loops": 1,
            "max_tokens": 1000,
            "temperature": 0.5,
        },
        "task": (
            "Radiology report: Chest X-ray shows mild cardiomegaly, no infiltrates, no effusion. Summarize the findings."
        ),
    },
]

result = client.agent.batch.run(body=batch_requests)

print(json.dumps(result, indent=4))
