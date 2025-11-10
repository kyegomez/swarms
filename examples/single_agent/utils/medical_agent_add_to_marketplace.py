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
            "description": (
                "Analyze blood samples and provide a report on the results, "
                "highlighting significant deviations, clinical context, red flags, "
                "and referencing established guidelines for lab test interpretation."
            ),
        },
        {
            "title": "Longitudinal Patient Lab Monitoring",
            "description": (
                "Process serial blood test results for a patient over time to identify clinical trends in key parameters (e.g., "
                "progression of anemia, impact of pharmacologic therapy, signs of organ dysfunction). Generate structured summaries "
                "that succinctly track rises, drops, or persistently abnormal markers. Flag patterns that suggest evolving risk or "
                "require physician escalation, such as a dropping platelet count, rising creatinine, or new-onset hyperglycemia. "
                "Report should distinguish true trends from ordinary biological variability, referencing clinical guidelines for "
                "critical-change thresholds and best-practice follow-up actions."
            ),
        },
        {
            "title": "Preoperative Laboratory Risk Stratification",
            "description": (
                "Interpret pre-surgical laboratory panels as part of risk assessment for patients scheduled for procedures. Identify "
                "abnormal or borderline values that may increase the risk of perioperative complications (e.g., bleeding risk from "
                "thrombocytopenia, signs of undiagnosed infection, electrolyte imbalances affecting anesthesia safety). Structure the "
                "output to clearly separate routine findings from emergent concerns, and suggest evidence-based adjustments, further "
                "workup, or consultation needs before proceeding with surgery, based on current clinical best practices and guideline "
                "recommendations."
            ),
        },
    ],
)

out = blood_analysis_agent.run(
    task="Analyze this blood sample: Hematology and Basic Metabolic Panel"
)

print(json.dumps(out, indent=4))
