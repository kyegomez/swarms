import os
from swarms_client import SwarmsClient
from dotenv import load_dotenv
import json

load_dotenv()

client = SwarmsClient(
    api_key=os.getenv("SWARMS_API_KEY"),
)


result = client.agent.run(
    agent_config={
        "agent_name": "Bloodwork Diagnosis Expert",
        "description": "An expert doctor specializing in interpreting and diagnosing blood work results.",
        "system_prompt": (
            "You are an expert medical doctor specializing in the interpretation and diagnosis of blood work. "
            "Your expertise includes analyzing laboratory results, identifying abnormal values, "
            "explaining their clinical significance, and recommending next diagnostic or treatment steps. "
            "Provide clear, evidence-based explanations and consider differential diagnoses based on blood test findings."
        ),
        "model_name": "groq/moonshotai/kimi-k2-instruct",
        "max_loops": 1,
        "max_tokens": 1000,
        "temperature": 0.5,
    },
    task=(
        "A patient presents with the following blood work results: "
        "Hemoglobin: 10.2 g/dL (low), WBC: 13,000 /µL (high), Platelets: 180,000 /µL (normal), "
        "ALT: 65 U/L (high), AST: 70 U/L (high). "
        "Please provide a detailed interpretation, possible diagnoses, and recommended next steps."
    ),
)

print(json.dumps(result, indent=4))
