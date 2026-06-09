# Group Chat Examples

This directory contains examples demonstrating the dynamic `GroupChat` — an asynchronous multi-agent room where every agent listens in parallel and independently decides whether to respond.

## Examples

- [enhanced_collaboration_example.py](enhanced_collaboration_example.py) - Analyst / researcher / strategist scenarios
- [medical_panel_example.py](medical_panel_example.py) - Panel of medical specialists discussing a complex multi-system case
- [quantum_physics_swarm.py](quantum_physics_swarm.py) - Condensed-matter physics research team
- [test_groupchat_feat.py](test_groupchat_feat.py) - Minimal smoke test
- [stream_example.py](stream_example.py) - Single-agent token streaming demo

## Subdirectories

- [groupchat_examples/](groupchat_examples/) - Domain-specific group chat scenarios (crypto tax, investment advisory, mortgage panel)

## Overview

`GroupChat` is an asynchronous, self-selecting agent room. There is no turn order, no speaker selection function, and no fixed number of rounds. When a message is posted, every other agent independently evaluates the message via a forced `respond(score, message)` tool call and broadcasts a reply only when its self-rated desire to speak clears the configured `threshold`. The conversation ends when either `max_loops` total messages have been posted or no new message arrives for `idle_timeout` seconds.

Every participating agent must be configured with `tools_list_dictionary=[RESPOND_TOOL]` (imported from `swarms.structs.groupchat`) so that it returns the structured speaking decision the chat depends on.
