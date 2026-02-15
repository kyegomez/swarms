# Dynamic Skills Loader

    Reference documentation for `swarms.structs.dynamic_skills_loader`.

    ## Overview

    This module provides production utilities for `dynamic skills loader` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.dynamic_skills_loader import ...
    ```

    ## Public API

    - `DynamicSkillsLoader`: `load_relevant_skills`, `get_skill_names`, `get_similarity_scores`

    ## Quick Start

    ```python
    from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader

loader = DynamicSkillsLoader(skills_dir="./skills", similarity_threshold=0.25)
print(loader.get_skill_names("Analyze quarterly revenue trends"))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/dynamic_skills_loader_example.md`](../examples/dynamic_skills_loader_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
