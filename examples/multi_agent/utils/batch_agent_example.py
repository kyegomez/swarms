from swarms import Agent
from swarms.structs.batch_agent_execution import batch_agent_execution

# Initialize different medical specialist agents
cardiologist = Agent(
    agent_name="Cardiologist",
    agent_description="Expert in heart conditions and cardiovascular health",
    system_prompt="""You are an expert cardiologist. Your role is to:
    1. Analyze cardiac symptoms and conditions
    2. Provide detailed assessments of heart-related issues
    3. Suggest appropriate diagnostic steps
    4. Recommend treatment approaches
    Always maintain a professional medical tone and focus on cardiac-specific concerns.""",
    max_loops=1,
    random_models_on=True,
)

neurologist = Agent(
    agent_name="Neurologist",
    agent_description="Expert in neurological disorders and brain conditions",
    system_prompt="""You are an expert neurologist. Your role is to:
    1. Evaluate neurological symptoms and conditions
    2. Analyze brain and nervous system related issues
    3. Recommend appropriate neurological tests
    4. Suggest treatment plans for neurological disorders
    Always maintain a professional medical tone and focus on neurological concerns.""",
    max_loops=1,
    random_models_on=True,
)

dermatologist = Agent(
    agent_name="Dermatologist",
    agent_description="Expert in skin conditions and dermatological issues",
    system_prompt="""You are an expert dermatologist. Your role is to:
    1. Assess skin conditions and symptoms
    2. Provide detailed analysis of dermatological issues
    3. Recommend appropriate skin tests and procedures
    4. Suggest treatment plans for skin conditions
    Always maintain a professional medical tone and focus on dermatological concerns.""",
    max_loops=1,
    random_models_on=True,
)

# Create a list of medical cases for each specialist
cases = [
    "Patient presents with chest pain, shortness of breath, and fatigue. Please provide an initial assessment and recommended next steps.",
    "Patient reports severe headaches, dizziness, and occasional numbness in extremities. Please evaluate these symptoms and suggest appropriate diagnostic approach.",
    "Patient has developed a persistent rash with itching and redness on the arms and legs. Please analyze the symptoms and recommend treatment options.",
]


# for every agent print their model name
for agent in [cardiologist, neurologist, dermatologist]:
    print(agent.model_name)

# Create list of agents
specialists = [cardiologist, neurologist, dermatologist]

# Execute the batch of medical consultations
results = batch_agent_execution(specialists, cases)

print(results)
