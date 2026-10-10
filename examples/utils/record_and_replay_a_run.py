"""Record a Swarms agent run once, then replay it offline as many times as you like.

Why: iterating on a prompt, a tool, or a system message normally means paying for the
whole run again on every change. Recording the run once and replaying it costs nothing
after the first pass, and makes "did my change alter the behaviour?" a cheap question.

How it works: Swarms routes model calls through LiteLLM, and LiteLLM honours both
`OPENAI_API_BASE` and `OPENAI_BASE_URL`. A recorder that moves one of those variables for
the child process therefore captures the run without any change to this file.

Verified on swarms 15.0.2 / litellm 1.76.1: with either variable pointed at a local
endpoint, `Agent(...).run(...)` sends `POST /v1/chat/completions` there and returns that
endpoint's answer.

Usage — record once, then replay:

    pip install swarms
    npm i -g orcareplay          # https://github.com/Continuum-AI-Corp/OrcaReplay

    orca record generic-openai -- python record_and_replay_a_run.py   # real run, recorded
    orca replay last                                                 # same run, no model called

    # Replay the first 4 steps from the recording, then continue on another model.
    # Everything before the fork is byte-identical, so the model is the only variable.
    orca replay last --from 4 --model claude-haiku-4-5

What replay does *not* tell you:

* A matching replay is not a determinism result. It proves the recorded exchange
  reproduces, not that the agent or the model is deterministic, and not that a fresh live
  run would behave the same way.
* Replay blocks model-provider egress only. It is not a sandbox: any tool the agent calls
  still runs for real on replay.
* Embedding calls are not captured by the default adapter, so a run whose behaviour
  depends on retrieval can replay cleanly at the model layer and still not reproduce.

Nothing below is recorder-specific — this is an ordinary Swarms agent.
"""

from swarms import Agent

agent = Agent(
    agent_name="Research-Assistant",
    agent_description="Answers a question and shows its reasoning",
    model_name="gpt-4o-mini",
    max_loops=1,
)

out = agent.run("List three tradeoffs of using a vector database, one line each.")
print(out)
