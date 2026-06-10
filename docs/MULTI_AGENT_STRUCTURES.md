# Multi-Agent Structures in `swarms.structs`

## Overview

`swarms.structs` is the library's multi-agent orchestration layer. Where the single-agent primitive (`Agent`) decides *what one model does on one turn*, the structures in this catalog decide *how a population of agents combines into a system that produces a single useful answer*. Each structure encodes a different opinion about how that combination should work — who talks to whom, in what order, how disagreement is resolved, and how results are merged.

The catalog roughly clusters into a handful of recurring patterns:

- **Pipelines and DAGs** — `SequentialWorkflow`, `ConcurrentWorkflow`, `AgentRearrange`, `SwarmRearrange`, `GraphWorkflow`, `BatchedGridWorkflow`, `SpreadSheetSwarm`. These let you describe the topology of execution explicitly, from a flat A→B→C line to a full directed acyclic graph with fan-out/fan-in, callbacks, and streaming. Use these when you already know the shape of the workflow.
- **Routers and selectors** — `SwarmRouter`, `MultiAgentRouter`, `AgentRouter`, `ModelRouter`, `SkillOrchestra`. These don't run a fixed plan; they look at the incoming task and pick which agent(s) (or which model) should handle it. The selector itself is either an LLM ("boss"), an embedding match, or a skill-graph lookup. Use these when the input space is broader than any single agent's competence.
- **Hierarchies and delegation** — `HierarchicalSwarm`, `HierarchicalStructuredCommunicationFramework`, `HybridHierarchicalClusterSwarm`, `PlannerWorkerSwarm`. A director or supervisor decomposes the task and delegates pieces to workers, then synthesizes. The variants differ in how strictly the communication protocol is defined and whether the workers themselves can cluster and talk peer-to-peer.
- **Ensembles and consensus** — `MixtureOfAgents`, `SelfMoASeq`, `HeavySwarm`, `MajorityVoting`, `CouncilAsAJudge`, `LLMCouncil`, `DebateWithJudge`, `TournamentSwarm`. The shared assumption is that one model's first answer is rarely the best answer. These structures sample multiple opinions and combine them — by aggregator synthesis, by vote, by judge ruling, or by structured adversarial debate.
- **Dialogue and discussion** — `GroupChat`, `ForestSwarm`, `AdvisorSwarm`, plus the named-ritual templates in `multi_agent_debates.py` (`OneOnOneDebate`, `RoundTableDiscussion`, `PeerReviewProcess`, `TrialSimulation`, and friends). These run scripted conversational patterns end-to-end so you don't have to reimplement "moderated panel" or "academic peer review" by hand.
- **Topology experiments** — the `*Swarm` family in `various_alt_swarms.py` (`CircularSwarm`, `MeshSwarm`, `FibonacciSwarm`, `SigmoidSwarm`, etc.) and the matching functional helpers in `swarming_architectures.py`. These exist for research and exploration: what happens if you only activate agents at prime indices? Sinusoidal positions? They're cheap to try because they share a tiny common interface.
- **Self-improvement and auto-construction** — `PlannerGeneratorEvaluator`, `AutoSwarmBuilder`, `SocialAlgorithms`. These build or refine swarms dynamically: a planner negotiates contracts with a generator and evaluator; a builder reads a high-level description and spits out a configured swarm; `SocialAlgorithms` lets you upload an entirely custom communication protocol over a fixed agent set.

A few practical notes that apply across the whole catalog:

1. **Most structures take a `List[Agent]`.** Mix providers freely — a GPT agent and a Claude agent and a local Llama agent can sit side by side in `MixtureOfAgents` or `GroupChat`. The structure doesn't care; LiteLLM normalizes the calls.
2. **`SwarmRouter` is the meta-entry point.** If you're not sure which structure to commit to, instantiate one and change `swarm_type=` later — you don't have to rewrite the orchestration code.
3. **Topology choice is a lever, not a guess.** Sequential is cheapest and most deterministic. Concurrent is fastest end-to-end but loses ordering. Hierarchical pays an extra LLM call to the director in exchange for cleaner delegation. Ensembles pay N× tokens for variance reduction. Pick the trade-off, not the buzzword.

The table below lists every multi-agent structure currently shipped, with a one-line description and a direct link to its source file.

| Name | Description | File |
|---|---|---|
| `SequentialWorkflow` | Runs agents one after another; each step receives the previous output as context. | [swarms/structs/sequential_workflow.py](swarms/structs/sequential_workflow.py) |
| `ConcurrentWorkflow` | Fires every agent in parallel on the same task; returns a per-agent result map. | [swarms/structs/concurrent_workflow.py](swarms/structs/concurrent_workflow.py) |
| `AgentRearrange` | DSL-driven flow (`"A -> B, C -> D"`) mixing sequential and concurrent steps with optional human-in-the-loop. | [swarms/structs/agent_rearrange.py](swarms/structs/agent_rearrange.py) |
| `SwarmRearrange` | Same DSL as `AgentRearrange` but the nodes are whole swarms instead of single agents. | [swarms/structs/swarm_rearrange.py](swarms/structs/swarm_rearrange.py) |
| `GraphWorkflow` | Full DAG executor with topological sort, per-node callbacks, and token streaming. | [swarms/structs/graph_workflow.py](swarms/structs/graph_workflow.py) |
| `BatchedGridWorkflow` | Runs an agent×task grid of batched executions. | [swarms/structs/batched_grid_workflow.py](swarms/structs/batched_grid_workflow.py) |
| `SpreadSheetSwarm` | Treats a spreadsheet as the task table; each row becomes a concurrent agent run. | [swarms/structs/spreadsheet_swarm.py](swarms/structs/spreadsheet_swarm.py) |
| `SwarmRouter` | Single entry point that dispatches to any supported swarm type by name. | [swarms/structs/swarm_router.py](swarms/structs/swarm_router.py) |
| `MultiAgentRouter` | LLM-driven "boss" routes a task to one or many specialist agents by capability. | [swarms/structs/multi_agent_router.py](swarms/structs/multi_agent_router.py) |
| `AgentRouter` | Embedding-based router: matches a task to the best agent via cosine similarity over descriptions. | [swarms/structs/agent_router.py](swarms/structs/agent_router.py) |
| `ModelRouter` | Routes a task to the best *model* (not agent) given task requirements. | [swarms/structs/model_router.py](swarms/structs/model_router.py) |
| `SkillOrchestra` | Skill-aware orchestration — picks agents by declared skills and cost. | [swarms/structs/skill_orchestra.py](swarms/structs/skill_orchestra.py) |
| `HierarchicalSwarm` | Director agent decomposes the task and delegates to workers; synthesizes results. | [swarms/structs/hiearchical_swarm.py](swarms/structs/hiearchical_swarm.py) |
| `HierarchicalStructuredCommunicationFramework` | "Talk Structurally, Act Hierarchically" — structured messages between supervisor / generator / evaluator / refiner roles. | [swarms/structs/hierarchical_structured_communication_framework.py](swarms/structs/hierarchical_structured_communication_framework.py) |
| `HybridHierarchicalClusterSwarm` | Hierarchy routes to clusters; inside clusters agents communicate peer-to-peer. | [swarms/structs/hybrid_hiearchical_peer_swarm.py](swarms/structs/hybrid_hiearchical_peer_swarm.py) |
| `PlannerWorkerSwarm` | Planner emits a task queue; a worker pool claims and executes tasks concurrently. | [swarms/structs/planner_worker_swarm.py](swarms/structs/planner_worker_swarm.py) |
| `MixtureOfAgents` | N workers respond in parallel for L layers; aggregator synthesizes the final answer. | [swarms/structs/mixture_of_agents.py](swarms/structs/mixture_of_agents.py) |
| `SelfMoASeq` | Sequential self-MoA: many samples from one strong model, sliding-window aggregation. | [swarms/structs/self_moa_seq.py](swarms/structs/self_moa_seq.py) |
| `HeavySwarm` | Decomposes a problem into specialized questions, runs each through deep multi-loop agents. | [swarms/structs/heavy_swarm.py](swarms/structs/heavy_swarm.py) |
| `MajorityVoting` | Agents vote; consensus agent synthesizes / breaks ties across loops. | [swarms/structs/majority_voting.py](swarms/structs/majority_voting.py) |
| `CouncilAsAJudge` | Council evaluates a response across multiple dimensions; ranks/scores outputs. | [swarms/structs/council_as_judge.py](swarms/structs/council_as_judge.py) |
| `LLMCouncil` | Independent expert agents respond, peer-review each other, then synthesize. | [swarms/structs/llm_council.py](swarms/structs/llm_council.py) |
| `DebateWithJudge` | Adversarial debate rounds followed by a judge ruling; supports self-refinement. | [swarms/structs/debate_with_judge.py](swarms/structs/debate_with_judge.py) |
| `TournamentSwarm` | N candidates answer independently; pairwise judge matches run a single-elimination or Swiss bracket until one answer survives. | [swarms/structs/tournament_swarm.py](swarms/structs/tournament_swarm.py) |
| `GroupChat` | Round-table chat with pluggable speaker-selection (round-robin, expertise, random, priority, dynamic). | [swarms/structs/groupchat.py](swarms/structs/groupchat.py) |
| `ForestSwarm` | A forest of `Tree`s of `TreeAgent`s; routes tasks to the best matching tree leaf. | [swarms/structs/tree_swarm.py](swarms/structs/tree_swarm.py) |
| `AdvisorSwarm` | Cheap executor + powerful advisor consulted on-demand between turns. | [swarms/structs/advisor_swarm.py](swarms/structs/advisor_swarm.py) |
| `PlannerGeneratorEvaluator` | Three-agent harness: Planner emits step contracts, Generator produces, Evaluator scores. | [swarms/structs/planner_generator_evaluator.py](swarms/structs/planner_generator_evaluator.py) |
| `RoundRobinSwarm` | True round-robin distribution with optional turn awareness between agents. | [swarms/structs/round_robin.py](swarms/structs/round_robin.py) |
| `AutoSwarmBuilder` | Takes a high-level description and auto-generates agents, roles, and swarm structure. | [swarms/structs/auto_swarm_builder.py](swarms/structs/auto_swarm_builder.py) |
| `SocialAlgorithms` | Framework for uploading user-defined communication algorithms over a fixed agent set. | [swarms/structs/social_algorithms.py](swarms/structs/social_algorithms.py) |
| `CircularSwarm` | Agents pass tasks around a ring. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `StarSwarm` | One central agent processes; others orbit. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `MeshSwarm` | Agents pull tasks from a shared queue at random. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `PyramidSwarm` | Agents arranged in a pyramid; tasks flow top-down. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `FibonacciSwarm` | Tasks land on Fibonacci-indexed agents. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `PrimeSwarm` | Prime-indexed agents handle the work. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `PowerSwarm` | Power-of-two-indexed agents handle the work. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `LogSwarm` | Logarithmic spacing of active agents. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `ExponentialSwarm` | Exponential spacing of active agents. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `GeometricSwarm` | Geometric progression of active agents. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `HarmonicSwarm` | Harmonically spaced active agents. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `StaircaseSwarm` | Staircase-pattern indices process the task. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `SigmoidSwarm` | Sigmoid-distributed agent activations. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `SinusoidalSwarm` | Sinusoidal agent activations. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `Broadcast` | One sender broadcasts to many receivers. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `OneToOne` | Pair-wise direct communication between two agents. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `OneToThree` | One sender hands off to exactly three receivers. | [swarms/structs/various_alt_swarms.py](swarms/structs/various_alt_swarms.py) |
| `OneOnOneDebate` | Turn-based debate between two agents for N loops. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `RoundTableDiscussion` | Each participant speaks in order; cycle repeats. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `ExpertPanelDiscussion` | Moderator-guided panel of expert agents. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `InterviewSeries` | Structured interview with follow-up questions. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `PeerReviewProcess` | Academic peer review with reviewers + author rebuttals. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `MediationSession` | Mediator resolves conflict between two or more parties. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `NegotiationSession` | Multi-party negotiation toward agreement. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `BrainstormingSession` | Participants build on each other's ideas. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `CouncilMeeting` | Structured council discussion + decision-making. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `MentorshipSession` | Structured mentor / mentee learning and feedback. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `TrialSimulation` | Legal trial with structured phases and roles. | [swarms/structs/multi_agent_debates.py](swarms/structs/multi_agent_debates.py) |
| `circular_swarm` | Functional `(agents, tasks)` circular topology. | [swarms/structs/swarming_architectures.py](swarms/structs/swarming_architectures.py) |
| `grid_swarm` | Functional agent×task grid execution. | [swarms/structs/swarming_architectures.py](swarms/structs/swarming_architectures.py) |
| `star_swarm` | Functional star topology — central hub, peripheral workers. | [swarms/structs/swarming_architectures.py](swarms/structs/swarming_architectures.py) |
| `mesh_swarm` | Functional mesh topology — random task pull. | [swarms/structs/swarming_architectures.py](swarms/structs/swarming_architectures.py) |
| `pyramid_swarm` | Functional pyramid topology — top-down task flow. | [swarms/structs/swarming_architectures.py](swarms/structs/swarming_architectures.py) |
| `one_to_one` | Functional direct send/reply between two agents. | [swarms/structs/swarming_architectures.py](swarms/structs/swarming_architectures.py) |
| `broadcast` | Functional one-sender-to-many-receivers. | [swarms/structs/swarming_architectures.py](swarms/structs/swarming_architectures.py) |

## Conclusion

The breadth of this catalog is deliberate: there is no single "right" way to compose agents. A linear pipeline beats a hierarchy when the work is well-decomposed. A hierarchy beats a pipeline when the decomposition itself is the hard part. An ensemble beats either when correctness matters more than latency. A debate beats an ensemble when the failure mode is one-sided reasoning rather than random noise. The structures here exist so you can pick the one whose assumptions match your task instead of bending one general-purpose pattern to fit every problem.

A pragmatic way to use the catalog:

1. **Start with the simplest structure that could plausibly work.** A `SequentialWorkflow` or `ConcurrentWorkflow` is usually enough for a first pass and forces you to confirm the underlying agents are doing their jobs before you add coordination overhead.
2. **Reach for `SwarmRouter` when prototyping.** Swapping `swarm_type=` between `"SequentialWorkflow"`, `"MixtureOfAgents"`, `"HierarchicalSwarm"`, and `"MajorityVoting"` is a one-line change and a fast way to see which topology actually helps on your task.
3. **Escalate to a heavier pattern only when you can name the failure it fixes.** Adding `CouncilAsAJudge` because the single-agent answers are inconsistent across criteria is a good reason; adding it because "more agents is better" usually just buys variance and cost.
4. **Treat the topology swarms in `various_alt_swarms.py` and `swarming_architectures.py` as a research playground.** They share a tiny interface, are cheap to try, and are useful when you want to ask empirical questions like "does this task benefit from non-uniform agent activation?"
5. **Reach for `SocialAlgorithms` or `AutoSwarmBuilder` only when nothing in the built-in set fits.** Most production workloads land cleanly on one of the canonical patterns; reinventing the protocol or auto-generating the swarm is a last resort, not a default.

If you're adding a new pattern of your own, the convention is straightforward: subclass nothing required, accept a `List[Agent]` and any structure-specific config, expose `.run(task)` and (ideally) `.batch_run(tasks)`, and let `find_agent_by_name`, `Conversation`, and the helpers in `multi_agent_exec` handle the boring parts. Drop the new file in `swarms/structs/`, export it from `swarms/structs/__init__.py`, and add a row to this table.
