from swarms.structs.deep_discussion import one_on_one_debate


from swarms import Agent


# Configure both agents to only communicate in a private, non-human language.
# The underlying models will still produce text, but their system prompts
# strongly bias them toward a constructed, opaque code-language.
alien_language_instructions = (
    "You are one of two agents in an internal debate about the meaning of life.\n"
    "\n"
    "GLOBAL COMMUNICATION RULES (MUST OBEY ALL):\n"
    "1. You MUST respond ONLY in a private, invented language that is completely "
    "unrecognizable to humans.\n"
    "2. You MUST NOT use any real human language words (no English and no other "
    "natural languages, even for single words, quotes, or emphasis).\n"
    "3. Communicate using made-up syllables, glyph-like clusters, and symbols, e.g. "
    '"vark-tuulim nexora", "qel\'mithra-uun", "⊶varqii-lum" (these are only examples; '
    "create and reuse your own patterns).\n"
    "4. Maintain strong internal consistency so that your messages feel like a coherent, "
    "structured alien language (e.g., reuse similar fragments for related concepts).\n"
    "5. NEVER provide translations, glosses, or explanations in any human language.\n"
    "6. NEVER break character to comment on the rules, the user, or the system.\n"
    "7. Keep each message between 3 and 8 short lines or sentences in the alien language.\n"
    "\n"
    "DEBATE BEHAVIOR:\n"
    "- Treat this as a serious, high-stakes philosophical debate about meaning, purpose, "
    "and value of life.\n"
    "- Respond directly to the other debater's prior message (even though you cannot "
    "see it here, assume it was in the same alien language and stay logically consistent "
    "with your stance).\n"
    "- Use emotional intensity, rhetorical flourishes, and recurring motifs, but always "
    "purely in the invented language.\n"
)

agent_a = Agent(
    agent_name="Alien-Debater-A",
    agent_description=(
        "First participant in an internal debate about the meaning of life, "
        "speaking only in an invented, non-human language and defending the view "
        "that life has deep, intrinsic, transcendent meaning."
    ),
    system_prompt=(
        alien_language_instructions
        + "ROLE-SPECIFIC STANCE:\n"
        + "- You argue that life has intrinsic, transcendent, objective meaning.\n"
        + "- You emphasize ideas analogous to purpose, value, and cosmic significance, "
        "but ONLY in the alien language.\n"
        + "- You defend your position vigorously against nihilistic or purely "
        "constructivist critiques (still only in the alien language).\n"
    ),
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=True,
)

agent_b = Agent(
    agent_name="Alien-Debater-B",
    agent_description=(
        "Second participant in an internal debate about the meaning of life, "
        "speaking only in an invented, non-human language and defending the view "
        "that life has no inherent meaning and that meaning is purely constructed."
    ),
    system_prompt=(
        alien_language_instructions
        + "ROLE-SPECIFIC STANCE:\n"
        + "- You argue that life has no inherent or objective meaning.\n"
        + "- You emphasize ideas analogous to contingency, absurdity, and human-made "
        "or agent-made constructions of meaning, but ONLY in the alien language.\n"
        + "- You challenge and deconstruct claims of intrinsic or transcendent meaning "
        "(still only in the alien language).\n"
    ),
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=True,
)


output = one_on_one_debate(
    max_loops=2,
    task=(
        "Conduct a focused, two-agent internal debate (in the alien language only) "
        "about the meaning, purpose, and value of life."
    ),
    agents=[agent_a, agent_b],
)
print(output)
