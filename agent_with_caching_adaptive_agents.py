
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

from reflection_tuner import ReflectionTuner
import requests

def duckduckgo_search(query):
    # Simple DuckDuckGo search function for Data-Collector agent
    url = f"https://api.duckduckgo.com/?q={query}&format=json&pretty=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("AbstractText", "No data found")
    return "Failed to retrieve data"

# Initialize Planner and Data-Collector agents with DuckDuckGo search capability
planner_agent = Agent(
    agent_name="Delaware-C-Corp-Planner-Agent",
    system_prompt="Develop a quarterly strategic roadmap for a Delaware C Corp.",
    agent_description="Creates detailed plans and schedules.",
    llm=model,
    max_loops=2,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="delaware_c_corp_planner_agent.json",
    user_name="planner_user",
    retry_attempts=2,
    context_length=150000,
    long_term_memory=memory_sustainability,  # Reuse memory for demonstration purposes
)

data_collector_agent = Agent(
    agent_name="Delaware-C-Corp-Data-Collector-Agent",
    system_prompt="Collect and synthesize information from DuckDuckGo search.",
    agent_description="Gathers data from open-source search engines.",
    llm=model,
    max_loops=3,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="delaware_c_corp_data_collector_agent.json",
    user_name="data_collector_user",
    retry_attempts=3,
    context_length=200000,
    long_term_memory=memory_risk,  # Reuse memory for demonstration
)

# Initialize Reflection Tuners
planner_reflection_tuner = ReflectionTuner(planner_agent, reflection_steps=2)
data_collector_reflection_tuner = ReflectionTuner(data_collector_agent, reflection_steps=2)

# Run Planner agent with Reflection Tuning
planner_response = planner_reflection_tuner.reflect_and_tune("Create a quarterly strategic roadmap for a Delaware C Corp in biotech.")
print("Planner Agent Response:", planner_response)

# Run Data Collector agent with Reflection Tuning, using DuckDuckGo search
data_collector_task = "Find recent trends in tax strategies for corporations in the US."
search_result = duckduckgo_search(data_collector_task)
data_collector_response = data_collector_reflection_tuner.reflect_and_tune(f"{search_result}")
print("Data Collector Agent Response:", data_collector_response)

from token_cache_and_adaptive_factory import TokenCache, AdaptiveAgentFactory

# Initialize TokenCache and AdaptiveAgentFactory
token_cache = TokenCache(cache_duration_minutes=30)  # Cache duration for tokens
adaptive_factory = AdaptiveAgentFactory(model, token_cache)

# Example of creating adaptive agents dynamically
adaptive_risk_agent = adaptive_factory.create_agent(
    agent_name="Adaptive-Risk-Agent",
    system_prompt="Assess new risk factors for changing economic conditions.",
    task="Dynamic risk analysis in evolving markets.",
    memory=memory_risk,
)

adaptive_sustainability_agent = adaptive_factory.create_agent(
    agent_name="Adaptive-Sustainability-Agent",
    system_prompt="Evaluate sustainability strategies in response to new regulations.",
    task="Dynamic sustainability strategy for manufacturing.",
    memory=memory_sustainability,
)

# Running adaptive agents
adaptive_risk_response = adaptive_risk_agent.run("Analyze potential economic risks for new market conditions.")
adaptive_sustainability_response = adaptive_sustainability_agent.run("Evaluate ESG strategies with upcoming regulation changes.")

print("Adaptive Risk Agent Response:", adaptive_risk_response)
print("Adaptive Sustainability Agent Response:", adaptive_sustainability_response)
