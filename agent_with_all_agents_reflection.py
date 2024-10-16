
from swarms import Agent
from swarm_models import OpenAIChat
from swarms_memory import ChromaDB
import os

# Initialize memory for agents
memory_risk = ChromaDB(metric="cosine", output_dir="risk_analysis_results")
memory_sustainability = ChromaDB(metric="cosine", output_dir="sustainability_results")

# Initialize model
model = OpenAIChat(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini", temperature=0.1)

# Initialize Risk Analysis Agent
risk_analysis_agent = Agent(
    agent_name="Delaware-C-Corp-Risk-Analysis-Agent",
    system_prompt="You are a specialized risk analysis agent focused on assessing risks.",
    agent_description="Performs risk analysis for Delaware C Corps.",
    llm=model,
    max_loops=3,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="delaware_c_corp_risk_analysis_agent.json",
    user_name="risk_analyst_user",
    retry_attempts=2,
    context_length=200000,
    long_term_memory=memory_risk,
)

# Initialize Sustainability Agent
sustainability_agent = Agent(
    agent_name="Delaware-C-Corp-Sustainability-Agent",
    system_prompt="You are a sustainability analysis agent focused on ESG factors.",
    agent_description="Analyzes sustainability practices for Delaware C Corps.",
    llm=model,
    max_loops=2,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=False,
    saved_state_path="delaware_c_corp_sustainability_agent.json",
    user_name="sustainability_specialist",
    retry_attempts=3,
    context_length=180000,
    long_term_memory=memory_sustainability,
)

# Run the agents
risk_analysis_agent.run("What are the top financial and operational risks for a Delaware C Corp in healthcare?")
sustainability_agent.run("How can a Delaware C Corp in manufacturing improve its sustainability practices?")

from reflection_tuner import ReflectionTuner

# Initialize Reflection Tuners for each agent
risk_reflection_tuner = ReflectionTuner(risk_analysis_agent, reflection_steps=2)
sustainability_reflection_tuner = ReflectionTuner(sustainability_agent, reflection_steps=2)

# Run the agents with Reflection Tuning
risk_response = risk_reflection_tuner.reflect_and_tune("What are the top financial and operational risks for a Delaware C Corp in healthcare?")
sustainability_response = sustainability_reflection_tuner.reflect_and_tune("How can a Delaware C Corp in manufacturing improve its sustainability practices?")

print("Risk Analysis Agent Response:", risk_response)
print("Sustainability Agent Response:", sustainability_response)

# Initialize agents from agents_with_new.yaml
# Import ReflectionTuner
from reflection_tuner import ReflectionTuner

# Initialize Reflection Tuner for all agents, including existing ones
deduction_agent = Agent(
    agent_name="Delaware-C-Corp-Tax-Deduction-Agent",
    system_prompt="Provide expert advice on tax deductions for Delaware C Corps.",
    agent_description="Analyzes tax deduction strategies.",
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="delaware_c_corp_tax_deduction_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=250000,
    long_term_memory=memory_risk,  # Reuse memory for testing purposes
)

optimization_agent = Agent(
    agent_name="Delaware-C-Corp-Tax-Optimization-Agent",
    system_prompt="Provide expert advice on tax optimization strategies for Delaware C Corps.",
    agent_description="Analyzes tax optimization.",
    llm=model,
    max_loops=2,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=False,
    saved_state_path="delaware_c_corp_tax_optimization_agent.json",
    user_name="tax_optimization_user",
    retry_attempts=3,
    context_length=200000,
    long_term_memory=memory_risk,
)

# Initialize Reflection Tuners
deduction_reflection_tuner = ReflectionTuner(deduction_agent, reflection_steps=2)
optimization_reflection_tuner = ReflectionTuner(optimization_agent, reflection_steps=2)

# Run agents with Reflection Tuning
deduction_response = deduction_reflection_tuner.reflect_and_tune("What are the most effective tax deduction strategies for a Delaware C Corp in tech?")
optimization_response = optimization_reflection_tuner.reflect_and_tune("How can a Delaware C Corp in finance optimize its tax strategy?")

print("Tax Deduction Agent Response:", deduction_response)
print("Tax Optimization Agent Response:", optimization_response)
