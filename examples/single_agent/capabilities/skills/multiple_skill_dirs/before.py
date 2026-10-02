"""
Skills from two directories, before #2395.

``skills_dir`` took a single directory, and passing a list raised
``TypeError``. To give an agent the team's skills and your own, you copied
both folders into one merged directory. The copy goes stale as soon as
either source folder changes.

Compare with after.py, which passes both directories directly.
"""

import shutil
import tempfile
from pathlib import Path

from swarms import Agent

here = Path(__file__).parent
merged = Path(tempfile.mkdtemp(prefix="skills_"))

for source in [here / "team_skills", here / "my_skills"]:
    shutil.copytree(source, merged, dirs_exist_ok=True)

agent = Agent(
    agent_name="Code-Reviewer",
    model_name="gpt-5.4",
    max_loops=1,
    skills_dir=str(merged),
)

print("Skills available:")
for skill in agent.load_skills_metadata():
    print(f"  {skill['name']:<16} {skill['path']}")

response = agent.run(
    "Review this Python code for bugs, missing tests and security issues:\n\n"
    "def get_user(db, name):\n"
    "    return db.execute(f\"SELECT * FROM users WHERE name = '{name}'\")"
)
print(response)
