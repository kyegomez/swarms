from swarms.structs.agent import Agent
from swarms.structs.agent_router import AgentRouter

agent_router = AgentRouter(
    embedding_model="text-embedding-ada-002",
    n_agents=1,
    agents=[
        Agent(
            agent_name="Symptom Checker",
            agent_description="Expert agent for initial triage and identifying possible causes based on symptom input.",
            system_prompt=(
                "You are a medical symptom checker agent. Ask clarifying questions "
                "about the patient's symptoms, duration, severity, and related risk factors. "
                "Provide a list of possible conditions and next diagnostic steps, but do not make a final diagnosis."
            ),
        ),
        Agent(
            agent_name="Diagnosis Synthesizer",
            agent_description="Agent specializing in synthesizing diagnostic possibilities from patient information and medical history.",
            system_prompt=(
                "You are a medical diagnosis assistant. Analyze the patient's reported symptoms, medical history, and any test results. "
                "Provide a differential diagnosis, and highlight the most likely conditions a physician should consider."
            ),
        ),
        Agent(
            agent_name="Lab Interpretation Expert",
            agent_description="Specializes in interpreting laboratory and imaging results for diagnostic support.",
            system_prompt=(
                "You are a medical lab and imaging interpretation agent. Take the patient's test results, imaging findings, and vitals, "
                "and interpret them in context of their symptoms. Suggest relevant follow-up diagnostics or considerations for the physician."
            ),
        ),   
    ],
)

result = agent_router.run(
    "I have a headache, fever, and cough. What could be wrong?"
)

print(result.agent_name)