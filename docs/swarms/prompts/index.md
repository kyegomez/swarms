# Prompts

The `prompts` package provides reusable system prompts and prompt-management primitives for building agents quickly.

## Public API

From `swarms.prompts`:

- `Prompt`
- `CODE_INTERPRETER`
- `DOCUMENTATION_WRITER_SOP`
- `FINANCE_AGENT_PROMPT`
- `GROWTH_AGENT_PROMPT`
- `LEGAL_AGENT_PROMPT`
- `OPERATIONS_AGENT_PROMPT`
- `PRODUCT_AGENT_PROMPT`
- `AUTONOMOUS_AGENT_SYSTEM_PROMPT`
- `get_autonomous_agent_prompt(...)`
- `get_autonomous_agent_prompt_with_context(...)`

## Prompt Model

`Prompt` is a versioned prompt object with:

- Content and metadata (`name`, `description`, `content`)
- Edit tracking (`edit_count`, `edit_history`)
- Rollback support (`rollback(version)`)
- Optional autosave to workspace (`autosave=True`)
- Tool schema injection (`add_tools(tools)`)

## Quickstart

```python
from swarms.prompts import Prompt

prompt = Prompt(
    name="research-agent",
    description="Research system prompt",
    content="You are a precise research assistant.",
    autosave=False,
)

prompt.edit_prompt("You are a precise research assistant. Always cite assumptions.")
print(prompt.get_prompt())
```

## Prompt Library Coverage

The package also includes many domain prompt modules (finance, legal, operations, education, multimodal, safety, orchestration, and more). Import directly from `swarms.prompts.<module>` when you need specialized templates.

## Notes

- Use `Prompt` when prompts need lifecycle/version control.
- Use static constants for simple immutable system prompts.
- Validate prompt updates in staging before production rollout.
