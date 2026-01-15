import json

from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader

loader = DynamicSkillsLoader(
    skills_dir="./agent_skill_examples", similarity_threshold=0.4
)

print(
    json.dumps(
        loader.load_relevant_skills("Create data visualizations"),
        indent=4,
    )
)
 