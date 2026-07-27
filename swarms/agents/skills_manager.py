"""
Agent Skills manager.

This module owns every piece of Agent Skills behaviour that used to be inlined
in ``swarms/structs/agent.py``:

* discovery and Tier 1 metadata loading of ``SKILL.md`` files (name +
  description + body, parsed from YAML frontmatter)
* static loading — every skill in ``skills_dir`` is folded into the system
  prompt
* dynamic loading — only the skills whose description is similar to the task
  are folded in, via :class:`DynamicSkillsLoader`
* Tier 2 loading — ``load_full_skill`` pulls the complete body of a single
  skill on demand
* rendering the skills section that gets appended to an agent's system prompt

``SkillsManager`` never mutates the agent. It returns prompt text; the caller
decides what to do with it. That keeps prompt mutation in one visible place in
``Agent`` and makes this class testable on its own.
"""

import os
from typing import Dict, List, Optional

import yaml
from loguru import logger

from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader

SKILL_FILENAME = "SKILL.md"

SKILLS_PROMPT_HEADER = (
    "\n\n# Available Skills\n\n"
    "You have access to the following specialized skills. "
    "Follow their instructions when relevant:\n\n"
)


class SkillsManager:
    """
    Loads Agent Skills from disk and renders them into system-prompt text.

    Implements the tiered loading model from Anthropic's Agent Skills
    framework: Tier 1 keeps name/description metadata in memory for
    context-aware activation, Tier 2 loads a skill's full body only when it is
    actually needed.

    Args:
        skills_dir: Directory containing skill folders. Each folder should hold
            a ``SKILL.md`` file with YAML frontmatter. ``None`` disables skills
            entirely.
        similarity_threshold: Minimum task/skill similarity for a skill to be
            selected during dynamic loading.

    Attributes:
        skills_dir (Optional[str]): The configured skills directory.
        metadata (List[Dict[str, str]]): Metadata for the skills loaded so far.

    Example:
        >>> skills = SkillsManager(skills_dir="./skills")
        >>> prompt_section = skills.prompt_for_task("Build a DCF model")
        >>> agent.system_prompt += prompt_section
    """

    def __init__(
        self,
        skills_dir: Optional[str] = None,
        similarity_threshold: float = 0.3,
    ):
        self.skills_dir = skills_dir
        self.similarity_threshold = similarity_threshold
        self.metadata: List[Dict[str, str]] = []
        self.dynamic_loader: Optional[DynamicSkillsLoader] = None

    def set_skills_dir(self, skills_dir: Optional[str]) -> None:
        """
        Point the manager at a different skills directory.

        Discards anything cached for the previous directory.
        """
        self.skills_dir = skills_dir
        self.dynamic_loader = None
        self.metadata = []

    @property
    def enabled(self) -> bool:
        """True when a usable skills directory is configured."""
        return bool(self.skills_dir) and os.path.exists(
            self.skills_dir
        )

    def prompt_for_task(self, task: Optional[str] = None) -> str:
        """
        Build the skills prompt section for a task.

        Args:
            task: Task description. When provided, only skills similar to the
                task are loaded. When ``None``, every skill is loaded.

        Returns:
            Formatted prompt section, or ``""`` when nothing was loaded.
        """
        if not self.skills_dir:
            return ""

        if task is not None:
            return self._dynamic_prompt(task)

        return self._static_prompt()

    def _static_prompt(self) -> str:
        """Load every skill in ``skills_dir`` and render it."""
        self.metadata = self.load_metadata()

        if not self.metadata:
            return ""

        logger.info(
            f"Loaded {len(self.metadata)} skills from {self.skills_dir}"
        )
        return self.build_prompt(self.metadata)

    def _dynamic_prompt(self, task: str) -> str:
        """Load only the skills relevant to ``task`` and render them."""
        if self.dynamic_loader is None:
            self.dynamic_loader = DynamicSkillsLoader(
                self.skills_dir,
                similarity_threshold=self.similarity_threshold,
            )

        logger.info(
            f"Loading dynamic skills for task: {task[:100]}..."
        )

        relevant_skills = self.dynamic_loader.load_relevant_skills(
            task
        )

        if not relevant_skills:
            logger.info(
                f"No relevant skills found for task: {task[:100]}..."
            )
            return ""

        self.metadata = relevant_skills
        logger.info(
            f"Dynamically loaded {len(relevant_skills)} relevant skills "
            f"for task: {task[:100]}..."
        )
        return self.build_prompt(relevant_skills)

    def load_metadata(
        self, skills_dir: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Load skill metadata from ``SKILL.md`` files (Tier 1 loading).

        Args:
            skills_dir: Directory to scan. Defaults to the configured
                ``skills_dir``.

        Returns:
            List of dicts with ``name``, ``description``, ``path`` and
            ``content`` keys. Empty when the directory is missing.
        """
        skills_dir = skills_dir or self.skills_dir
        skills: List[Dict[str, str]] = []

        if not skills_dir or not os.path.exists(skills_dir):
            logger.warning(
                f"Skills directory not found: {skills_dir}"
            )
            return skills

        for skill_folder in sorted(os.listdir(skills_dir)):
            skill_path = os.path.join(skills_dir, skill_folder)

            if not os.path.isdir(skill_path):
                continue

            skill_file = os.path.join(skill_path, SKILL_FILENAME)

            if not os.path.exists(skill_file):
                continue

            skill = self._parse_skill_file(skill_file, skill_folder)

            if skill is not None:
                skills.append(skill)

        return skills

    def _parse_skill_file(
        self, skill_file: str, fallback_name: str
    ) -> Optional[Dict[str, str]]:
        """
        Parse a single ``SKILL.md`` into a metadata dict.

        Returns ``None`` when the file has no YAML frontmatter or cannot be
        read — a malformed skill is skipped, never fatal.
        """
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.startswith("---"):
                return None

            parts = content.split("---", 2)

            if len(parts) < 3:
                return None

            frontmatter = yaml.safe_load(parts[1]) or {}
            name = frontmatter.get("name", fallback_name)

            logger.info(f"Loaded skill: {name}")

            return {
                "name": name,
                "description": frontmatter.get("description", ""),
                "path": skill_file,
                "content": parts[2].strip(),
            }
        except Exception as e:
            logger.warning(
                f"Failed to load skill from {skill_file}: {e}"
            )
            return None

    def build_prompt(self, skills: List[Dict[str, str]]) -> str:
        """
        Render skill metadata as a system-prompt section.

        Args:
            skills: Metadata dicts from :meth:`load_metadata`.

        Returns:
            Formatted prompt section, or ``""`` when ``skills`` is empty.
        """
        if not skills:
            return ""

        prompt = SKILLS_PROMPT_HEADER

        for skill in skills:
            prompt += f"## {skill['name']}\n\n"
            prompt += f"**Description**: {skill['description']}\n\n"
            prompt += skill["content"]
            prompt += "\n\n---\n\n"

        return prompt

    def load_full_skill(self, skill_name: str) -> Optional[str]:
        """
        Load the complete body of one skill (Tier 2 loading).

        Args:
            skill_name: Name of the skill, as it appears in :attr:`metadata`.

        Returns:
            The markdown below the frontmatter, or ``None`` when the skill is
            unknown or unreadable.
        """
        for skill in self.metadata:
            if skill["name"] != skill_name:
                continue

            try:
                with open(skill["path"], "r", encoding="utf-8") as f:
                    content = f.read()

                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        return parts[2].strip()
            except Exception as e:
                logger.error(
                    f"Failed to load full skill {skill_name}: {e}"
                )

        return None
