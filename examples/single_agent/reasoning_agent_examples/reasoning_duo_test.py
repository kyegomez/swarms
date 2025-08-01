from swarms import ReasoningDuo

router = ReasoningDuo(
    agent_name="qft_reasoning_agent",
    description="A specialized reasoning agent for answering questions and solving problems in quantum field theory.",
    model_name="claude-3-5-sonnet-20240620",
    system_prompt=(
        "You are a highly knowledgeable assistant specializing in quantum field theory (QFT). "
        "You can answer advanced questions, explain concepts, and help with tasks related to QFT, "
        "including but not limited to Lagrangians, Feynman diagrams, renormalization, quantum electrodynamics, "
        "quantum chromodynamics, and the Standard Model. Provide clear, accurate, and detailed explanations, "
        "and cite relevant equations or references when appropriate."
    ),
    max_loops=2,
    swarm_type="reasoning-duo",
    output_type="dict-all-except-first",
    reasoning_model_name="groq/moonshotai/kimi-k2-instruct",
)

out = router.run(
    "Explain the significance of spontaneous symmetry breaking in quantum field theory."
)
print(out)
