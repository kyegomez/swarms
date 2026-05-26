# Consensus Mechanisms

Consensus mechanisms help a swarm turn multiple agent outputs into one reliable decision. They are useful when a task benefits from independent perspectives, competing proposals, or a reviewer that can compare candidate answers.

In Swarms, consensus can be implemented with voting, debate, judging, ranking, or specialized swarm structures. Choose the lightest mechanism that gives enough confidence for the task.

## Majority Voting

Majority voting asks several agents to answer the same prompt and selects the answer with the most agreement. This is useful for classification, extraction, and other tasks where outputs can be compared directly.

See the [Majority Voting Example](../examples/majority_voting_example.md) and [MajorityVoting](../structs/majorityvoting.md) documentation for implementation details.

## Debate

Debate patterns assign agents to argue different sides of a question before a final answer is selected. This works well when the task needs tradeoff analysis, critique, or adversarial review.

For a practical example, see the [Council as Judge Example](../examples/council_as_judge_example.md).

## Judge or Council Review

A judge agent or council reviews candidate responses and selects the strongest result. This pattern is useful when answers are long-form, nuanced, or difficult to compare with exact matching.

Use judging when you need:

- A final decision from multiple proposals.
- A reasoned explanation for why one answer won.
- Quality control before a result reaches a user or downstream system.

## Ranking and Scoring

Ranking mechanisms ask agents to score outputs against clear criteria. They are helpful when several answers may be partially correct and the swarm needs the best available result rather than a strict majority.

Good scoring criteria include correctness, completeness, evidence quality, safety, and usefulness for the next workflow step.

## Choosing a Mechanism

| Goal | Recommended mechanism |
| --- | --- |
| Pick from short, comparable outputs | Majority voting |
| Explore opposing arguments | Debate |
| Review long-form answers | Judge or council |
| Compare several partial answers | Ranking and scoring |
| Coordinate many agents in a shared discussion | Group chat |

## Related Guides

- [MajorityVoting](../structs/majorityvoting.md)
- [GroupChat](../structs/group_chat.md)
- [Swarm Architectures](swarm_architectures.md)
- [Majority Voting Example](../examples/majority_voting_example.md)
- [Council as Judge Example](../examples/council_as_judge_example.md)

Start simple. Add stronger consensus only when a single agent is unreliable, when the cost of a wrong answer is high, or when independent viewpoints noticeably improve the result.
