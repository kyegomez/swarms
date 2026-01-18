"""Planner configuration: model selection and prompts."""

# Minimal config for model selection per role
MODEL_SELECTION = {
    "primary": {
        "model": "gpt-5.2",
        "context_window": 200_000,
    },
    "subplanner": {"model": "gpt-5.1", "context_window": 50_000},
}

# Example prompts (placeholders). In practice these are tuned per project.
PROMPTS = {
    "primary_explore": "You are a primary planner: explore the codebase and suggest high-level tasks.",
    "sub_breakdown": "You are a sub-planner: break down a task into smaller actionable items.",
}
