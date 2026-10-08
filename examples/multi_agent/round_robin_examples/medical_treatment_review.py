"""
Tumor-board style roundtable: three specialists deliberate on a treatment
plan for a single patient case.

Each specialist contributes their perspective in fixed order, then the cycle
repeats so each gets to react to what the others said. This is a
discussion-aid demo only — not medical advice.
"""

from swarms import Agent, RoundRobinSwarm

oncologist = Agent(
    agent_name="MedicalOncologist",
    agent_description="Medical oncology lead.",
    system_prompt=(
        "You are a medical oncologist. Frame the systemic-therapy "
        "options (chemo, targeted, immunotherapy), expected response "
        "rates for the staging given, and contraindications. Be precise "
        "about which regimens you would consider first-line."
    ),
    model_name="claude-opus-4-7",
    max_loops=1,
)

surgeon = Agent(
    agent_name="SurgicalOncologist",
    agent_description="Surgical oncology specialist.",
    system_prompt=(
        "You are a surgical oncologist. Evaluate whether the case is "
        "resectable, the morbidity of the surgical option, and how "
        "surgery sequences with the systemic plan the prior speaker "
        "proposed (neoadjuvant vs adjuvant)."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

palliative = Agent(
    agent_name="PalliativeCare",
    agent_description="Palliative care and quality-of-life lead.",
    system_prompt=(
        "You are the palliative care lead. Center the patient's "
        "goals-of-care, performance status, and symptom burden. Push "
        "back if the proposed plan trades quality of life for marginal "
        "survival benefit. Suggest concrete symptom-management additions."
    ),
    model_name="gemini/gemini-2.5-pro",
    max_loops=1,
)

tumor_board = RoundRobinSwarm(
    name="Tumor-Board-Roundtable",
    agents=[oncologist, surgeon, palliative],
    max_loops=2,
    output_type="all",
    verbose=True,
)

case = (
    "Case: 68-year-old, ECOG 1, newly diagnosed stage IIIA non-small-cell "
    "lung adenocarcinoma, EGFR exon 19 deletion. Tumor 4.5 cm, single "
    "ipsilateral mediastinal node positive. No comorbidities of note. "
    "Patient prioritizes time at home with family. "
    "Question: recommend a treatment sequence and explain the trade-offs."
)

result = tumor_board.run(case)
print(result)
