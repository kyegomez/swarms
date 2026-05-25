# Swarms Improvement Proposals

Swarms Improvement Proposals, or SIPs, are lightweight design documents for
changes that need shared agreement before implementation. They help contributors
explain the motivation, expected behavior, and migration impact of larger
changes before code or documentation is split across several pull requests.

## When To Write An SIP

Write an SIP for:

- New public agent, swarm, or workflow abstractions.
- Changes to existing public APIs or configuration behavior.
- Cross-repository integrations that need a stable contract.
- Major documentation reorganizations or new contributor programs.
- Changes that require coordinated examples, tests, and migration guidance.

You do not need an SIP for small bug fixes, typo corrections, narrow examples,
or documentation pages that clarify existing behavior.

## Suggested Template

```markdown
# SIP: Short Title

## Summary
One or two paragraphs explaining the proposed change.

## Motivation
Why this change matters and which users it helps.

## Detailed Design
The API, behavior, documentation structure, or process being proposed.

## Compatibility
Expected migration impact, deprecations, or backwards-compatibility notes.

## Alternatives
Other options considered and why this proposal is preferred.

## Rollout Plan
Implementation steps, documentation updates, tests, and follow-up work.
```

## Submission Checklist

- Link the SIP to the related issue, discussion, or pull request.
- Keep the first version concise enough for maintainers to review quickly.
- Call out breaking changes explicitly.
- Include examples when the proposal changes developer-facing behavior.
- Update the SIP after review so the accepted decision is easy to find later.
