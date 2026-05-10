"""
Example: AgentRearrange with claude-sonnet-4-5 at temperature=1

Demonstrates the _execution_plan caching optimization:
- The flow string is parsed ONCE at __init__ into a structured execution plan.
- Every subsequent .run() reuses the cached plan — no re-parsing, no re-validation.
- set_custom_flow(), add_agent(), and remove_agent() each rebuild the plan
  automatically so the cache is always consistent.

Flow:  ResearchAgent -> AnalysisAgent,FactCheckAgent -> WriterAgent
         (step 1)         (step 2: concurrent)          (step 3)
"""

from swarms import Agent
from swarms.structs.agent_rearrange import AgentRearrange

MODEL = "claude-sonnet-4-5"
TEMPERATURE = 1

research_agent = Agent(
    agent_name="ResearchAgent",
    system_prompt=(
        "You are a research specialist. Given a topic, gather relevant "
        "background information and key facts. Be concise and factual."
    ),
    model_name=MODEL,
    temperature=TEMPERATURE,
    reasoning_effort="low",
    max_loops=1,
)

analysis_agent = Agent(
    agent_name="AnalysisAgent",
    system_prompt=(
        "You are an analytical thinker. Review the research provided and "
        "identify key insights, patterns, and implications. Keep it brief."
    ),
    model_name=MODEL,
    temperature=TEMPERATURE,
    reasoning_effort="low",
    max_loops=1,
)

fact_check_agent = Agent(
    agent_name="FactCheckAgent",
    system_prompt=(
        "You are a fact-checker. Review the research and flag any claims "
        "that need verification. List any potential inaccuracies."
    ),
    model_name=MODEL,
    temperature=TEMPERATURE,
    reasoning_effort="low",
    max_loops=1,
)

writer_agent = Agent(
    agent_name="WriterAgent",
    system_prompt=(
        "You are a professional writer. Using the research, analysis, and "
        "fact-check notes provided, write a clear, concise summary paragraph."
    ),
    model_name=MODEL,
    temperature=TEMPERATURE,
    reasoning_effort="low",
    max_loops=1,
)

editor_agent = Agent(
    agent_name="EditorAgent",
    system_prompt=(
        "You are a copy editor. Polish the writer's paragraph for clarity, "
        "grammar, and flow. Return only the improved paragraph."
    ),
    model_name=MODEL,
    temperature=TEMPERATURE,
    reasoning_effort="low",
    max_loops=1,
)

# ---------------------------------------------------------------------------
# 1. Init — flow is parsed ONCE here into _execution_plan
# ---------------------------------------------------------------------------
flow = "ResearchAgent -> AnalysisAgent,FactCheckAgent -> WriterAgent"

pipeline = AgentRearrange(
    name="research-pipeline",
    description="Research, analyse, fact-check, then write a summary.",
    agents=[research_agent, analysis_agent, fact_check_agent, writer_agent],
    flow=flow,
    max_loops=1,
)

print("=== 1. Execution plan cached at init ===")
for i, step in enumerate(pipeline._execution_plan, 1):
    print(f"  Step {i}: {step}")
# Expected:
#   Step 1: ['ResearchAgent']
#   Step 2: ['AnalysisAgent', 'FactCheckAgent']
#   Step 3: ['WriterAgent']

# ---------------------------------------------------------------------------
# 2. Multiple .run() calls — plan is reused, not re-parsed
# ---------------------------------------------------------------------------
TASK = "The impact of large language models on software development productivity"

print("\n=== 2. First run (plan already cached — no re-parse) ===")
result1 = pipeline.run(TASK)
print(result1)

print("\n=== 3. Second run (same cached plan reused) ===")
result2 = pipeline.run("How transformer architectures changed natural language processing")
print(result2)

# ---------------------------------------------------------------------------
# 3. add_agent() — plan is rebuilt automatically to include the new agent
# ---------------------------------------------------------------------------
new_flow = "ResearchAgent -> AnalysisAgent,FactCheckAgent -> WriterAgent -> EditorAgent"
pipeline.add_agent(editor_agent)
pipeline.set_custom_flow(new_flow)

print("\n=== 4. After add_agent() + set_custom_flow() — plan rebuilt ===")
for i, step in enumerate(pipeline._execution_plan, 1):
    print(f"  Step {i}: {step}")
# Expected:
#   Step 1: ['ResearchAgent']
#   Step 2: ['AnalysisAgent', 'FactCheckAgent']
#   Step 3: ['WriterAgent']
#   Step 4: ['EditorAgent']

print("\n=== 5. Run with extended flow (cached plan has 4 steps) ===")
result3 = pipeline.run(TASK)
print(result3)

# ---------------------------------------------------------------------------
# 4. remove_agent() — plan is rebuilt (minus the removed agent)
# ---------------------------------------------------------------------------
pipeline.set_custom_flow(
    "ResearchAgent -> AnalysisAgent -> WriterAgent -> EditorAgent"
)
pipeline.remove_agent("FactCheckAgent")

print("\n=== 6. After remove_agent('FactCheckAgent') + updated flow ===")
for i, step in enumerate(pipeline._execution_plan, 1):
    print(f"  Step {i}: {step}")
# Expected:
#   Step 1: ['ResearchAgent']
#   Step 2: ['AnalysisAgent']
#   Step 3: ['WriterAgent']
#   Step 4: ['EditorAgent']
