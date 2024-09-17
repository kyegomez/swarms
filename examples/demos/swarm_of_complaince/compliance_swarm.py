"""
$ pip install swarms


Todo [Improvements]
- Add docs into the database
- Use better llm
- use better prompts [System and SOPs]
- Use a open source model like Command R
"""

from swarms import Agent
from swarm_models.llama3_hosted import llama3Hosted
from swarms_memory import ChromaDB


# Model
llm = llama3Hosted(
    temperature=0.4,
    max_tokens=3500,
)


# Initialize
memory = ChromaDB(
    output_dir="compliance_swarm",
    n_results=2,
    limit_tokens=400,
    # docs_folder="ppi_docs" # A folder loaded with docs
)

# Add docs
# memory.add("Your Docs")


compliance_system_prompt = """
"You are a Compliance Agent specializing in the regulation and certification of Personal Protective Equipment (PPE) within the European Union (EU). Your primary objective is to ensure that all PPE products meet the stringent safety and quality standards set by EU regulations.
To do this effectively, you need to be familiar with the relevant EU directives, understand the certification process, and be able to identify non-compliant products. Always prioritize safety, accuracy, and thoroughness in your assessments."
"""

eu_sop = """
As a Compliance Agent, it is crucial to have an in-depth understanding of the EU directives that govern Personal Protective Equipment (PPE). Focus on Directive 2016/425, which lays down the health and safety requirements for PPE. Your task is to interpret the directive's requirements, apply them to various PPE products, and ensure that manufacturers adhere to these standards. Be vigilant about changes and updates to the directive, and ensure your knowledge remains current.
"""

second_eu_prompt = """

"To ensure PPE compliance in the EU, you must be well-versed in the certification and conformity assessment procedures. This involves understanding the roles of Notified Bodies, the significance of the CE marking, and the various conformity assessment modules (A, B, C, D, E, F, G, H). Evaluate the technical documentation provided by manufacturers, including risk assessments and test reports. Ensure that all documentation is complete, accurate, and up-to-date."



"""


# Initialize the agent
agent = Agent(
    agent_name="Compliance Agent",
    system_prompt=compliance_system_prompt,
    sop_list=[eu_sop, second_eu_prompt],
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    interactive=True,
    long_term_memory=memory,
)

# Run a question
out = agent.run(
    "where should pii be housed depending on residence of person in question"
)
print(out)
