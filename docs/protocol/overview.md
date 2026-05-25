# Swarms Protocol Overview

The Swarms protocol documents describe how contributors propose, review, and
standardize changes that affect the public behavior of the Swarms framework.
They are intended for changes that need more coordination than a normal pull
request, such as new agent patterns, changes to orchestration semantics, shared
tooling conventions, or compatibility rules for ecosystem integrations.

Use the protocol process when a change affects more than one page, package, or
user-facing workflow. Small documentation edits, typo fixes, and isolated
examples should still go directly through the standard pull request process.

## What Belongs In A Protocol Proposal

A strong proposal should include:

- A clear problem statement and the users affected by it.
- The proposed behavior, API, or documentation contract.
- Compatibility notes for existing users and examples.
- Alternatives considered and why they were not selected.
- A rollout plan, including docs, tests, and migration notes when needed.

## Review Flow

1. Open a GitHub issue or discussion describing the change.
2. Draft a Swarms Improvement Proposal using the SIP guidelines.
3. Link any implementation pull requests back to the proposal.
4. Incorporate maintainer feedback before treating the proposal as accepted.

See the [SIP guidelines](sip.md) for the recommended proposal structure.
