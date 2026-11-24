# FileChannel examples

A `FileChannel` keeps the conversation in a markdown file. Every script here
writes its run to a file under `runs/`. Open the file while a script runs to
watch the agents talk, or edit it between runs to steer the next one.

Run any script from the repository root:

```bash
python examples/channel_examples/file_channel/sequential_workflow_channel.py
cat runs/sequential.md
```

| Script | What it shows | Writes to |
|---|---|---|
| [`agent_channel.py`](agent_channel.py) | Two lone agents sharing one file through `Agent(channel=...)` | `runs/ev_conversation.md` |
| [`sequential_workflow_channel.py`](sequential_workflow_channel.py) | A pipeline whose transcript lands in the file in step order | `runs/sequential.md` |
| [`concurrent_workflow_channel.py`](concurrent_workflow_channel.py) | A fan-out whose answers are posted in agent order, never interleaved | `runs/concurrent.md` |
| [`agent_rearrange_channel.py`](agent_rearrange_channel.py) | A flow with a parallel step, recorded by the structure | `runs/rearrange.md` |
| [`graph_workflow_channel.py`](graph_workflow_channel.py) | A DAG that posts each node's output as it completes | `runs/graph.md` |
| [`mixture_of_agents_channel.py`](mixture_of_agents_channel.py) | Workers plus an aggregator, with the synthesis as the last turn | `runs/mixture.md` |
| [`groupchat_channel.py`](groupchat_channel.py) | A self-selecting chat whose accepted messages become the file | `runs/groupchat.md` |
| [`hierarchical_swarm_channel.py`](hierarchical_swarm_channel.py) | A director's plan and each worker's report under their names | `runs/hierarchical.md` |
| [`planner_worker_swarm_channel.py`](planner_worker_swarm_channel.py) | A planner's steps and the workers that execute them | `runs/planner_worker.md` |
| [`majority_voting_channel.py`](majority_voting_channel.py) | Every vote and the consensus | `runs/majority.md` |
| [`swarm_router_channel.py`](swarm_router_channel.py) | One channel forwarded to whichever swarm the router builds | `runs/router.md` |
| [`resume_channel.py`](resume_channel.py) | A second run that continues an existing file; run `sequential_workflow_channel.py` first | `runs/sequential.md` |

Docs: [FileChannel](https://docs.swarms.world/api/channels/file-channel)
