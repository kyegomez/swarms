# Dynamic Skills Loader Tutorial

    End-to-end tutorial for `swarms.structs.dynamic_skills_loader`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader

loader = DynamicSkillsLoader(skills_dir="./skills", similarity_threshold=0.25)
print(loader.get_skill_names("Analyze quarterly revenue trends"))
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `dynamic_skills_loader`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/dynamic_skills_loader.md`](../structs/dynamic_skills_loader.md)
