# Dynamic Skills Loader

    `dynamic_skills_loader` reference documentation.

    **Module Path**: `swarms.structs.dynamic_skills_loader`

    ## Overview

    Task-similarity-driven skill loader that scores and selects relevant skills from `SKILL.md` metadata.

    ## Public API

    - **`DynamicSkillsLoader`**: `load_relevant_skills()`, `get_skill_names()`, `load_full_skill_content()`, `get_similarity_scores()`
- **`create_dynamic_skills_loader()`**

    ## Quickstart

    ```python
    from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader, create_dynamic_skills_loader
    ```

    ## Tutorial

    A runnable tutorial is available at [`swarms/examples/dynamic_skills_loader_example.md`](../examples/dynamic_skills_loader_example.md).

    ## Notes

    - Keep task payloads small for first runs.
    - Prefer deterministic prompts when comparing outputs across agents.
    - Validate provider credentials (for LLM-backed examples) before production use.
