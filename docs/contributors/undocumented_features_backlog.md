# Undocumented Features Backlog

This backlog tracks core framework features that currently do not have dedicated documentation pages under `docs/swarms/`.

## Issue Drafts and Branches

### 1) Payments policy engine docs

- **Issue title:** Document payments policy engine and spend controls
- **Issue:** #1350 — https://github.com/kyegomez/swarms/issues/1350
- **Branch name:** `docs/payments-policy-engine`
- **Scope:**
  - Explain `PolicyRule`, `Budget`, and `PolicyDecision`
  - Show `PolicyEngine.evaluate()` behavior and outcomes
  - Document `SpendTracker` windows (`hourly`, `daily`, `monthly`)
  - Add usage snippets for policy evaluation and spend recording

### 2) Telemetry docs

- **Issue title:** Document telemetry opt-in, payload shape, and environment toggles
- **Issue:** #1351 — https://github.com/kyegomez/swarms/issues/1351
- **Branch name:** `docs/telemetry-guide`
- **Scope:**
  - Explain telemetry entry points and helper functions
  - Document `SWARMS_TELEMETRY_ON` and `SWARMS_API_KEY`
  - Describe payload structure and endpoint behavior
  - Provide privacy and production hardening notes

### 3) Sims docs

- **Issue title:** Document simulation APIs with SenatorAssembly quickstart
- **Issue:** #1352 — https://github.com/kyegomez/swarms/issues/1352
- **Branch name:** `docs/sims-senator-assembly`
- **Scope:**
  - Explain available simulation exports
  - Add quickstart for `SenatorAssembly`
  - Document output and conversation flow patterns
  - Add scalability and cost considerations for large simulations

### 4) Artifacts docs

- **Issue title:** Document artifacts package and versioned file workflows
- **Issue:** #1356 — https://github.com/kyegomez/swarms/issues/1356
- **Branch name:** `docs/artifacts-guide`
- **Scope:**
  - Document `Artifact` and `FileVersion`
  - Explain create/edit/version/save/export lifecycle
  - Add quickstart for versioned content editing

### 5) Prompts docs

- **Issue title:** Document prompts package public API and lifecycle model
- **Issue:** #1357 — https://github.com/kyegomez/swarms/issues/1357
- **Branch name:** `docs/prompts-guide`
- **Scope:**
  - Document exported prompt constants and `Prompt` model
  - Explain edit history, rollback, and autosave behavior
  - Add quickstart for prompt authoring and updates

### 6) Schemas docs

- **Issue title:** Document schemas package for steps and MCP connections
- **Issue:** #1358 — https://github.com/kyegomez/swarms/issues/1358
- **Branch name:** `docs/schemas-guide`
- **Scope:**
  - Document `Step`, `ManySteps`, `MCPConnection`, `MultipleMCPConnections`
  - Explain common serialization/validation usage patterns
  - Add quickstart for MCP connection schema setup

## Current Status

- `payments`: Document page created at `docs/swarms/payments/index.md`
- `telemetry`: Document page created at `docs/swarms/telemetry/index.md`
- `sims`: Document page created at `docs/swarms/sims/index.md`
- `artifacts`: Document page created at `docs/swarms/artifacts/index.md`
- `prompts`: Document page created at `docs/swarms/prompts/index.md`
- `schemas`: Document page created at `docs/swarms/schemas/index.md`
- Local branches created:
  - `docs/payments-policy-engine`
  - `docs/telemetry-guide`
  - `docs/sims-senator-assembly`
  - `docs/artifacts-guide`
  - `docs/prompts-guide`
  - `docs/schemas-guide`
- GitHub issues created: #1350, #1351, #1352, #1356, #1357, #1358
