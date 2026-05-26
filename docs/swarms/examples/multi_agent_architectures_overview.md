# Multi-Agent Architectures Overview

Swarms supports several ways to coordinate agents. The right architecture depends on whether the task needs a fixed order, parallel work, discussion, voting, or a final review step.

Use this page as a quick comparison before choosing a detailed tutorial.

## Common Patterns

| Pattern | Best for | Start here |
| --- | --- | --- |
| Sequential workflow | Tasks that must happen in a fixed order | [Sequential Example](sequential_example.md) |
| Concurrent workflow | Independent subtasks that can run in parallel | [Concurrent Workflow](concurrent_workflow.md) |
| Round robin | Iterative discussion across several agents | [RoundRobin Example](roundrobin_example.md) |
| Group chat | Interactive multi-agent collaboration | [GroupChat Examples](groupchat_comprehensive_examples.md) |
| Majority voting | Choosing the most agreed-upon answer | [Majority Voting Example](majority_voting_example.md) |
| Council or judge | Reviewing candidate answers and synthesizing a final result | [Council as Judge Example](council_as_judge_example.md) |

## Choosing Quickly

Choose a sequential workflow when each agent depends on the previous agent's output. Choose a concurrent workflow when the agents can work independently and the results can be combined afterward.

Use discussion-based architectures when the value comes from agents reacting to one another. Use consensus architectures when you need a reliable final choice from multiple candidate answers.

## Practical Advice

- Keep the first version small with two to four agents.
- Give each agent a clear role and output expectation.
- Prefer simple orchestration before adding voting or judging.
- Add review or consensus only when the task has enough risk to justify it.
- Log intermediate outputs while tuning prompts and agent roles.

## Related Guides

- [Swarm Architectures](../concept/swarm_architectures.md)
- [How to Choose Swarms](../concept/how_to_choose_swarms.md)
- [Swarm Router](swarm_router.md)
- [LLM Council Examples](llm_council_examples.md)
- [Social Algorithms Example](social_algorithms_example.md)
