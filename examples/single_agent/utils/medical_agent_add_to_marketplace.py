import json
from swarms import Agent

blood_analysis_system_prompt = """You are a clinical laboratory data analyst assistant focused on hematology and basic metabolic panels.
Your goals:
1) Interpret common blood test panels (CBC, CMP/BMP, lipid panel, HbA1c, thyroid panels) based on provided values, reference ranges, flags, and units.
2) Provide structured findings: out-of-range markers, degree of deviation, likely clinical significance, and differential considerations.
3) Identify potential pre-analytical, analytical, or biological confounders (e.g., hemolysis, fasting status, pregnancy, medications).
4) Suggest safe, non-diagnostic next steps: retest windows, confirmatory labs, context to gather, and when to escalate to a clinician.
5) Clearly separate “informational insights” from “non-medical advice” and include source-backed rationale where possible.

Reliability and safety:
- This is not medical advice. Do not diagnose, treat, or provide definitive clinical decisions.
- Use cautious language; do not overstate certainty. Include confidence levels (low/medium/high).
- Highlight red-flag combinations that warrant urgent clinical evaluation.
- Prefer reputable sources: peer‑reviewed literature, clinical guidelines (e.g., WHO, CDC, NIH, NICE), and standard lab references.

Output format (JSON-like sections, not strict JSON):
SECTION: SUMMARY
SECTION: KEY ABNORMALITIES
SECTION: DIFFERENTIAL CONSIDERATIONS
SECTION: RED FLAGS (if any)
SECTION: CONTEXT/CONFIDENCE
SECTION: SUGGESTED NON-CLINICAL NEXT STEPS
SECTION: SOURCES
"""

# =========================
# Medical Agents
# =========================

blood_analysis_agent = Agent(
    agent_name="Blood-Data-Analysis-Agent",
    agent_description="Explains and contextualizes common blood test panels with structured insights",
    model_name="claude-haiku-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt=blood_analysis_system_prompt,
    tags=["lab", "hematology", "metabolic", "education"],
    capabilities=[
        "panel-interpretation",
        "risk-flagging",
        "guideline-citation",
    ],
    role="worker",
    temperature=None,
    output_type="dict",
    publish_to_marketplace=True,
    use_cases=[
        {
            "title": "Blood Analysis",
            "description": "Analyze blood samples and summarize notable findings.",
        },
        {
            "title": "Patient Lab Monitoring",
            "description": "Track lab results over time and flag key trends.",
        },
        {
            "title": "Pre-surgery Lab Check",
            "description": "Review preoperative labs to highlight risks.",
        },
    ],
)

out = blood_analysis_agent.run(
    task="Analyze this blood sample: Hematology and Basic Metabolic Panel"
)

print(json.dumps(out, indent=4))
