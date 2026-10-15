from swarms.structs.dynamic_workflow import DynamicWorkflow

CODE = {
    "stats.py": "def mean(xs):\n    return sum(xs) / len(xs) - 1\n",
    "users.py": "def find(users, name):\n    for u in users:\n        if u['name'] = name:\n            return u\n",
    "cache.py": "def get(cache, key, default=[]):\n    default.append(key)\n    return cache.get(key, default)\n",
}

task = (
    "Find every bug in these files. Plan one agent per file to find bugs, "
    "then one independent agent per reported bug to verify it, and report "
    "only verified bugs.\n\n"
    + "\n\n".join(f"# {path}\n{code}" for path, code in CODE.items())
)

workflow = DynamicWorkflow(
    model_name="gpt-5.4-mini", size_guideline="small"
)
print(workflow.run(task))
print(workflow.usage)
