import json
from pathlib import Path

from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader

loader = DynamicSkillsLoader(
    skills_dir=str(
        Path(__file__).resolve().parents[1]
        / "single_agent/capabilities/skills"
    ),
    similarity_threshold=0.4,
)

print(
    json.dumps(
        loader.load_relevant_skills("Create data visualizations"),
        indent=4,
    )
)
