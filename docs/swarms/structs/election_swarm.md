# Election Swarm

Election Swarm is a multi-agent decision pattern where candidates, reviewers, or voters evaluate options and select a winner. The architecture docs link to this route for users exploring voting-based swarm designs.

Use this page as a pattern reference. Current repository examples demonstrate election-style workflows in the examples tree.

## When to Use

- Selecting one option from several candidates
- Simulating board, committee, or panel decisions
- Comparing proposals through multiple voter perspectives
- Ranking outputs before a final decision
- Teaching voting and deliberation patterns with agents

## Pattern Shape

An election-style swarm usually includes:

- A list of candidates or proposals
- Voter agents with different criteria
- A ballot or scoring format
- A tallying step
- A final explanation of the winning result

## Example Task

```text
Review three product roadmap proposals.
Each voter should score the proposals on user impact,
technical risk, and revenue potential, then vote for one winner.
Return the final tally and reasoning.
```

## Related Docs

- [Swarm Architectures](../concept/swarm_architectures.md)
- [Majority Voting](majorityvoting.md)
- [GroupChat](group_chat.md)

## Source Examples

- `examples/multi_agent/election_swarm_examples/election_example.py`
- `examples/multi_agent/election_swarm_examples/apple_board_election_example.py`
