# Dynamic Skills Loader Tutorial

    End-to-end usage for `dynamic_skills_loader`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader

loader = DynamicSkillsLoader(skills_dir="./skills", similarity_threshold=0.25)

task = "Build a sales forecasting dashboard from quarterly data"
print(loader.get_skill_names(task))
print(loader.get_similarity_scores(task)[:3])
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `dynamic_skills_loader`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/dynamic_skills_loader.md`](../structs/dynamic_skills_loader.md)
