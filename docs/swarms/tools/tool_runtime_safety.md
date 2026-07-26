# Tool Runtime Safety and Validation Guide

Production Swarms agents often call tools that read files, call APIs, write records, or trigger downstream workflows. The tool interface is simple, but the runtime behavior can become risky when inputs are ambiguous, credentials are mixed with model-visible text, or a tool returns data that the next agent treats as trusted. This guide provides a practical safety checklist for building tools that are predictable, testable, and ready for agent workflows.

Use these practices when you expose a Python function through `Agent(tools=[...])`, convert functions with `BaseTool`, or wrap remote services as MCP tools. The goal is not to make every tool complicated. The goal is to make the tool contract explicit enough that an agent can use it correctly without guessing.

## 1. Start with a narrow contract

Every tool should do one job. Avoid tools named `run_task`, `process_request`, or `do_everything` because they force the model to infer behavior from a vague description. Prefer a precise name and a docstring that explains:

- what the tool does;
- what each argument expects;
- what it will never do;
- what the return value means;
- which failures are expected and recoverable.

```python
from typing import Literal


def summarize_invoice_total(
    currency: Literal["USD", "EUR"],
    line_items: list[dict[str, float]],
) -> dict[str, float | str]:
    """Return the subtotal for invoice line items in one supported currency.

    The tool only sums numeric `amount` fields. It does not charge a card,
    send an invoice, update accounting software, or infer exchange rates.
    """
    total = sum(float(item["amount"]) for item in line_items)
    return {"currency": currency, "total": round(total, 2)}
```

This small contract is easier for the agent to call and easier for reviewers to audit. If a workflow needs payment, email, and database writes, split those steps into separate tools and let the agent plan between them.

## 2. Validate before side effects

Tools that mutate state should validate every input before they create files, call external APIs, or write to a database. Treat arguments from an agent the same way you would treat arguments from a public API. Validate type, length, allowed values, and required fields first. Then perform the irreversible action.

A safe pattern is:

1. normalize input into a small internal structure;
2. reject unsupported values with a clear error;
3. perform read-only checks;
4. perform the side effect;
5. return a receipt that can be logged or shown to a human.

```python
def create_support_ticket(title: str, body: str, priority: str = "normal") -> dict[str, str]:
    """Create a support ticket after validating title, body, and priority."""
    allowed_priorities = {"low", "normal", "high"}

    if priority not in allowed_priorities:
        raise ValueError(f"priority must be one of {sorted(allowed_priorities)}")
    if not title.strip() or len(title) > 120:
        raise ValueError("title must be between 1 and 120 characters")
    if not body.strip() or len(body) > 4_000:
        raise ValueError("body must be between 1 and 4000 characters")

    # Replace this block with the real API call in production.
    ticket_id = "TICKET-123"
    return {"ticket_id": ticket_id, "status": "created", "priority": priority}
```

Clear errors are useful because the agent can retry with corrected input. Silent partial success is worse than a clean failure.

## 3. Keep secrets outside prompts and returns

Tools should read API keys, tokens, and private configuration from environment variables or a secret manager. Do not place secrets in tool descriptions, docstrings, return values, exception messages, or agent prompts. If a remote API returns a response that includes credentials, redact it before returning the data to the agent.

Recommended return format:

```python
{
    "ok": True,
    "resource_id": "job_123",
    "status": "queued",
    "redactions": ["api_token", "authorization_header"]
}
```

Avoid returning raw HTTP headers, full request bodies, or complete webhook payloads unless the workflow explicitly needs them. When in doubt, return a stable identifier and a short status summary.

## 4. Make output machine-readable

Agents are more reliable when tools return structured data instead of free-form paragraphs. A dictionary with stable keys lets another agent, router, or evaluator decide what to do next.

Good tool responses include:

- `ok`: whether the operation succeeded;
- `status`: a short status string such as `created`, `skipped`, or `needs_review`;
- `data`: the main result;
- `warnings`: non-fatal issues;
- `next_action`: the expected follow-up.

```python
def check_dataset_shape(rows: list[dict[str, object]]) -> dict[str, object]:
    """Inspect row count, fields, and missing values for a tabular dataset."""
    fields = sorted({key for row in rows for key in row.keys()})
    missing = {
        field: sum(1 for row in rows if row.get(field) in (None, ""))
        for field in fields
    }
    return {
        "ok": True,
        "status": "inspected",
        "data": {"rows": len(rows), "fields": fields, "missing": missing},
        "warnings": [
            f"{field} has {count} missing value(s)"
            for field, count in missing.items()
            if count
        ],
        "next_action": "review_warnings_before_import",
    }
```

## 5. Add dry-run modes for destructive tools

If a tool can delete, overwrite, publish, purchase, transfer, or notify someone, add a `dry_run` argument. A dry run should execute all validation and planning steps, then return what would happen without performing the action.

```python
def archive_old_reports(report_ids: list[str], dry_run: bool = True) -> dict[str, object]:
    """Archive selected reports, or preview the archive plan when dry_run is true."""
    if not report_ids:
        raise ValueError("report_ids cannot be empty")

    plan = [{"report_id": report_id, "action": "archive"} for report_id in report_ids]
    if dry_run:
        return {"ok": True, "status": "preview", "data": plan}

    # Perform the real archive operation here.
    return {"ok": True, "status": "archived", "data": plan}
```

Use `dry_run=True` as the default for any action that could surprise a user. Require the agent workflow to make an explicit second call when it is ready to commit.

## 6. Log receipts, not secrets

For long-running agent systems, logs are essential. Store the tool name, validated arguments, result status, resource identifiers, and timestamps. Do not store private tokens, raw credentials, or personal data unless the product has a clear retention policy.

A useful audit record is:

```python
{
    "tool": "create_support_ticket",
    "status": "created",
    "resource_id": "TICKET-123",
    "timestamp": "2026-01-29T10:00:00Z"
}
```

This is enough to debug the workflow without exposing sensitive data.

## 7. Review checklist

Before merging a new tool, confirm:

- the tool name describes a single action;
- the docstring explains limits and side effects;
- type hints cover every argument and return value;
- validation happens before side effects;
- errors are clear enough for an agent to correct;
- secrets never appear in prompts, logs, exceptions, or returns;
- outputs are dictionaries with stable keys;
- destructive tools support dry-run behavior;
- examples can be copied into a small test file;
- external APIs have timeout and retry policies.

These checks make Swarms tools easier to compose in multi-agent systems and safer to expose in production environments. They also make documentation examples easier to maintain because contributors can reason about the contract without reading the full implementation.
