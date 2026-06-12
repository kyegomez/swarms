"""SwarmRouter + GroupChat (v12.0.3).

- The speaker-function API is gone: instead of
  SwarmRouter(..., speaker_function=round_robin_speaker), just pick the
  GroupChat swarm type — agents decide for themselves when to speak.
- Bug fix: output_type and verbose are now forwarded to the underlying
  GroupChat instead of being silently dropped.
"""

from swarms import Agent, SwarmRouter

agents = [
    Agent(
        agent_name="Optimist",
        system_prompt="Argue the upside.",
        model_name="gpt-4.1",
        max_loops=1,
    ),
    Agent(
        agent_name="Pessimist",
        system_prompt="Argue the risks.",
        model_name="gpt-4.1",
        max_loops=1,
    ),
]

router = SwarmRouter(
    agents=agents,
    swarm_type="GroupChat",
    output_type="dict",  # both settings now actually reach the GroupChat
    verbose=True,
)

result = router.run("Should we adopt AI for medical diagnosis?")
print(result)
