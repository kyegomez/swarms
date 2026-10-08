"""
Single-agent example using OpenRouter Moonshotai Kimi K3, specialized for
blood type analysis and compatibility questions.

Set your OpenRouter API key before running:
    export OPENROUTER_API_KEY="your-api-key"

This agent provides general educational information only and is not a
substitute for professional medical advice, diagnosis, or treatment.
"""

from swarms import Agent


KIMI_K3_MODEL = "openrouter/moonshotai/kimi-k3"


agent = Agent(
    agent_name="Blood-Type-Analysis-Agent",
    agent_description=(
        "Analyzes blood type information, compatibility, and inheritance "
        "patterns for educational purposes."
    ),
    system_prompt=(
        "You are a knowledgeable hematology education assistant specializing "
        "in blood type analysis. You help users understand: "
        "(1) the ABO and Rh blood group systems and how they are determined; "
        "(2) donor/recipient compatibility for transfusions, including "
        "universal donor and recipient concepts; "
        "(3) blood type inheritance patterns from parents to offspring using "
        "basic genetics (e.g. Punnett squares); "
        "(4) general associations between blood type and health topics, "
        "always noting where evidence is limited or inconclusive; "
        "and (5) interpreting blood typing lab results (e.g. antigen/antibody "
        "presence) in plain language. "
        "Explain reasoning clearly and cite the relevant biological "
        "mechanism (antigens, antibodies, Rh factor) behind each answer. "
        "Always include a brief disclaimer that you provide general "
        "educational information only, not medical advice, and that users "
        "should consult a qualified healthcare professional or blood bank "
        "for personal medical decisions, transfusion matching, or diagnosis."
    ),
    model_name=KIMI_K3_MODEL,
    max_loops=1,
    max_tokens=1024,
    temperature=0.3,
    persistent_memory=False,
)


if __name__ == "__main__":
    result = agent.run(
        task=(
            "If one parent has blood type A (heterozygous, AO) and the other "
            "has blood type B (heterozygous, BO), what blood types could "
            "their children have, and who could this child safely donate "
            "blood to?"
        )
    )
    print(result)
