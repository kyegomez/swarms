# Swarms Improvement Proposals

Swarms Improvement Proposals (SIPs) are lightweight design proposals for major framework changes. Use a SIP when a change affects public APIs, agent orchestration behavior, configuration formats, integrations, or long-term project direction.

For small bug fixes, documentation updates, examples, or isolated implementation changes, open a normal GitHub issue or pull request instead.

## When to Open a SIP

Open a SIP for changes such as:

- New agent or swarm architecture types
- Changes to public constructor parameters or return values
- New protocol, tool, model-provider, or deployment integrations
- Behavior changes that affect existing user workflows
- Large documentation or migration efforts that need maintainer alignment

## Proposal Format

Use a concise issue or discussion with the following sections:

```markdown
## Summary
What should change?

## Motivation
Why is this needed, and who benefits?

## Proposed Design
How should the change work at a high level?

## Compatibility
Does this break existing users, examples, or configuration files?

## Alternatives
What other approaches were considered?

## Validation
How can maintainers and contributors test or review the change?
```

## Review Process

1. Search existing issues and discussions for related proposals.
2. Open a GitHub issue describing the SIP with the format above.
3. Wait for maintainer feedback before beginning a large implementation.
4. Keep implementation pull requests focused and link them back to the SIP.
5. Update relevant docs, examples, and tests as the design is accepted.

## Acceptance Criteria

A SIP is ready to implement when the maintainers agree on the problem, the proposed behavior, compatibility expectations, and the validation plan.
