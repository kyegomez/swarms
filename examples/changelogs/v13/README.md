# v13 Examples — Changelog Code Samples

Runnable versions of every code example from the root `CHANGELOG.md` (changes from May 5 to June 12, 2026, releases 12.0.1 → 12.0.3).

Set an API key before running any example:

```bash
export OPENAI_API_KEY="sk-..."
```

## New Features

| Example | Feature |
| --- | --- |
| [groupchat_self_selecting_example.py](groupchat_self_selecting_example.py) | Async self-selecting GroupChat with auto-injected respond tool |
| [graph_workflow_composition_example.py](graph_workflow_composition_example.py) | GraphWorkflow nested subgraphs, `max_parallel_nodes`, `validate(raise_on_error)` |
| [sequential_workflow_streaming_example.py](sequential_workflow_streaming_example.py) | SequentialWorkflow `run_stream` — plain tokens and structured events |
| [agent_rearrange_flow_example.py](agent_rearrange_flow_example.py) | Pure concurrent flows, `explain()`, output-type validation |
| [round_robin_rotation_example.py](round_robin_rotation_example.py) | True round-robin rotation with turn awareness |
| [heavy_swarm_variant_example.py](heavy_swarm_variant_example.py) | HeavySwarm `variant` parameter |
| [agent_lookup_helpers_example.py](agent_lookup_helpers_example.py) | `find_agent_by_id`, cached `find_agent_by_name`, `return_all_agent_names`, `to_dict()` |
| [cli_examples.sh](cli_examples.sh) | CLI tips, model discovery, typo correction |

## Improvements & Fixes

| Example | Change |
| --- | --- |
| [swarm_router_groupchat_example.py](swarm_router_groupchat_example.py) | Speaker-function API removed; `output_type`/`verbose` forwarded to GroupChat |
| [agent_tools_default_example.py](agent_tools_default_example.py) | `tools_list_dictionary` defaults to an empty list |
