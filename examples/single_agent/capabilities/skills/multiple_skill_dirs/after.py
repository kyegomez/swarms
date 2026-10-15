"""
Skills from two directories, after #2395.

Pass ``skills_dir`` a list. The agent gets ``team_skills/code-review`` and
``my_skills/commit-messages`` without copying either. Directories are read
in order, and when two of them hold a skill with the same name, the later
directory's skill is used and a warning names both locations.

Compare with before.py, which had to copy both folders into one.
"""

from pathlib import Path

from swarms import Agent

here = Path(__file__).parent

agent = Agent(
    agent_name="Code-Reviewer",
    model_name="gpt-5.4",
    max_loops=1,
    skills_dir=[str(here / "team_skills"), str(here / "my_skills")],
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
