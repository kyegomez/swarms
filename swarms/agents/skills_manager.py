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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import requests
import yaml
from loguru import logger

from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader
from swarms.utils.url_guard import is_safe_url

SKILL_FILENAME = "SKILL.md"

SKILL_FETCH_TIMEOUT = 30

MAX_SKILL_FETCH_WORKERS = 8

SKILLS_PROMPT_HEADER = (
    "\n\n# Available Skills\n\n"
    "You have access to the following specialized skills. "
    "Follow their instructions when relevant:\n\n"
)


@lru_cache(maxsize=64)
def _fetch_skill_markdown(url: str) -> str:
    """
    Fetch one skill's markdown, once per process.

    Args:
        url: Marketplace ``.md`` URL.

    Returns:
        The response body.

    Raises:
        ValueError: If the URL targets a private or link-local address.

    Notes:
        The guard sits inside the cache on purpose - a hit issues no request,
        and ``lru_cache`` does not memoize the raise, so a blocked URL is
        rejected on every call. Same shape as ``_fetch_image_url``.
    """
    if not is_safe_url(url):
        raise ValueError(
            f"Blocked skill URL '{url}': only external HTTP/HTTPS URLs are permitted."
        )

    response = requests.get(url, timeout=SKILL_FETCH_TIMEOUT)
    response.raise_for_status()
    return response.text


class SkillsManager:
    """
    Loads Agent Skills from disk and renders them into system-prompt text.

    Implements the tiered loading model from Anthropic's Agent Skills
    framework: Tier 1 keeps name/description metadata in memory for
    context-aware activation, Tier 2 loads a skill's full body only when it is
    actually needed.

    Args:
        skills_dir: Directory containing skill folders. Each folder should hold
            a ``SKILL.md`` file with YAML frontmatter. ``None`` disables local
            skills.
        skill_urls: Marketplace ``.md`` URLs, each serving one skill in the
            same frontmatter format. Fetched concurrently and rendered in the
            order given. Composes with ``skills_dir``.
        similarity_threshold: Minimum task/skill similarity for a skill to be
            selected during dynamic loading.

    Attributes:
        skills_dir (Optional[str]): The configured skills directory.
        skill_urls (List[str]): The configured remote skill URLs.
        metadata (List[Dict[str, str]]): Metadata for the skills loaded so far.

    Example:
        >>> skills = SkillsManager(skills_dir="./skills")
        >>> prompt_section = skills.prompt_for_task("Build a DCF model")
        >>> agent.system_prompt += prompt_section
    """

    def __init__(
        self,
        skills_dir: Optional[str] = None,
        skill_urls: Optional[Sequence[str]] = None,
        similarity_threshold: float = 0.3,
    ):
        self.skills_dir = skills_dir
        self.skill_urls: List[str] = list(skill_urls or [])
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

    def set_skill_urls(
        self, skill_urls: Optional[Sequence[str]]
    ) -> None:
        """
        Replace the remote skill URLs.

        Discards anything cached for the previous list.
        """
        self.skill_urls = list(skill_urls or [])
        self.metadata = []

    @property
    def enabled(self) -> bool:
        """True when a usable skill source is configured, local or remote."""
        if self.skill_urls:
            return True
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
        if not self.skills_dir and not self.skill_urls:
            return ""

        if task is not None:
            return self._dynamic_prompt(task)

        return self._static_prompt()

    def _static_prompt(self) -> str:
        """Load every configured skill and render it."""
        self.metadata = self._static_skills()

        if not self.metadata:
            return ""

        return self.build_prompt(self.metadata)

    def _dynamic_prompt(self, task: str) -> str:
        """Load only the skills relevant to ``task`` and render them."""
        self.metadata = self._dynamic_skills(task)

        if not self.metadata:
            return ""

        return self.build_prompt(self.metadata)

    def _static_skills(self) -> List[Dict[str, str]]:
        """Every local skill, then every remote one, in the order configured."""
        skills = self.load_metadata() if self.skills_dir else []

        if self.skill_urls:
            skills = skills + self.load_remote_metadata()

        logger.info(
            f"Loaded {len(skills)} skills "
            f"({self.skills_dir or 'no directory'}, "
            f"{len(self.skill_urls)} urls)"
        )
        return skills

    def _dynamic_skills(self, task: str) -> List[Dict[str, str]]:
        """Only the skills whose description is similar to ``task``."""
        loader = self._loader()

        logger.info(
            f"Loading dynamic skills for task: {task[:100]}..."
        )

        skills = (
            loader.load_relevant_skills(task)
            if self.skills_dir
            else []
        )

        if self.skill_urls:
            skills = skills + loader.select_relevant(
                task, self.load_remote_metadata()
            )

        if not skills:
            logger.info(
                f"No relevant skills found for task: {task[:100]}..."
            )
            return []

        logger.info(
            f"Dynamically loaded {len(skills)} relevant skills "
            f"for task: {task[:100]}..."
        )
        return skills

    def _loader(self) -> DynamicSkillsLoader:
        """The similarity loader, built once and reused."""
        if self.dynamic_loader is None:
            self.dynamic_loader = DynamicSkillsLoader(
                self.skills_dir or "",
                similarity_threshold=self.similarity_threshold,
            )
        return self.dynamic_loader

    def load_remote_metadata(self) -> List[Dict[str, str]]:
        """
        Load skill metadata from ``skill_urls`` (Tier 1 loading, remote).

        Returns:
            List of dicts with the same keys :meth:`load_metadata` returns,
            in the order the URLs were configured. A URL that cannot be
            fetched, or whose body carries no frontmatter, is skipped with a
            warning - one dead link does not stop the run.

        Notes:
            Order is preserved because the marketplace list is the caller's
            stated priority and skills render into the prompt in that order.
            Only approved marketplace prompts are served, so a valid, listed
            URL returning 404 is an ordinary case rather than an edge one.
        """
        if not self.skill_urls:
            return []

        workers = min(len(self.skill_urls), MAX_SKILL_FETCH_WORKERS)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            bodies = list(
                executor.map(self._fetch_skill, self.skill_urls)
            )

        skills: List[Dict[str, str]] = []
        for url, content in zip(self.skill_urls, bodies):
            if content is None:
                continue

            skill = self._parse_skill_markdown(content, url, url)

            if skill is None:
                logger.warning(
                    f"Skipping skill from {url}: no YAML frontmatter"
                )
                continue

            skills.append(skill)

        return skills

    def _fetch_skill(self, url: str) -> Optional[str]:
        """Fetch one skill URL, returning ``None`` when it cannot be read."""
        try:
            return _fetch_skill_markdown(url)
        except Exception as e:
            logger.warning(f"Failed to load skill from {url}: {e}")
            return None

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
        except Exception as e:
            logger.warning(
                f"Failed to load skill from {skill_file}: {e}"
            )
            return None

        return self._parse_skill_markdown(
            content, fallback_name, skill_file
        )

    def _parse_skill_markdown(
        self, content: str, fallback_name: str, source: str
    ) -> Optional[Dict[str, str]]:
        """
        Parse skill markdown into a metadata dict.

        Shared by the file and URL sources so both produce the same shape.

        Args:
            content: The full ``SKILL.md`` text, frontmatter included.
            fallback_name: Used when the frontmatter carries no ``name``.
            source: Where the text came from — a file path or a URL. Stored
                as ``path``, which is what Tier 2 loading looks up.

        Returns:
            A metadata dict, or ``None`` when there is no YAML frontmatter.

        Notes:
            ``split("---", 2)`` stops after the frontmatter, so horizontal
            rules in the body survive intact.

            Marketplace frontmatter also carries ``id``, ``source`` and
            ``created_at``. They are kept rather than dropped: ``id`` is a
            stable identity for dedup, ``source`` is provenance, and
            ``created_at`` is the hook for version pinning.
        """
        try:
            if not content.startswith("---"):
                return None

            parts = content.split("---", 2)

            if len(parts) < 3:
                return None

            frontmatter = yaml.safe_load(parts[1]) or {}
            name = frontmatter.get("name", fallback_name)

            logger.info(f"Loaded skill: {name}")

            skill = {
                "name": name,
                "description": frontmatter.get("description", ""),
                "path": source,
                "content": parts[2].strip(),
            }

            for extra in ("id", "source", "created_at"):
                if extra in frontmatter:
                    skill[extra] = str(frontmatter[extra])

            return skill
        except Exception as e:
            logger.warning(
                f"Failed to parse skill from {source}: {e}"
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

        Notes:
            A remote skill's ``path`` is a URL, not a file. Its body is
            already in memory from the Tier 1 fetch, so it is returned from
            there rather than fetched a second time.
        """
        for skill in self.metadata:
            if skill["name"] != skill_name:
                continue

            if skill["path"] in self.skill_urls:
                return skill["content"]

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
