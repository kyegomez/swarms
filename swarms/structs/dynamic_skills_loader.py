"""
Dynamic Skills Loader for Agent

This module provides intelligent skill loading based on task similarity.
It tokenizes and embeds both user tasks and skill descriptions, then uses
cosine similarity to determine which skills should be loaded into the agent.
"""

import math
import os
import re
from typing import Callable, Dict, List, Optional, Union

import yaml
from loguru import logger


def skill_dirs(
    skills_dir: Optional[Union[str, List[str]]],
) -> List[str]:
    """
    Normalise a ``skills_dir`` setting to a list of directories.

    Args:
        skills_dir: One directory, a list of directories, or ``None``.

    Returns:
        The directories in the order given.
    """
    if isinstance(skills_dir, str):
        return [skills_dir]
    return list(skills_dir or [])


def load_skill_dirs(
    skills_dir: Optional[Union[str, List[str]]],
    load_dir: Callable[[str], List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """
    Load the skills of every directory, in order.

    When two directories hold a skill with the same name, the skill from
    the later directory replaces the earlier one and a warning is logged.

    Args:
        skills_dir: One directory or a list of directories.
        load_dir: Loads the skills of a single directory.

    Returns:
        List of skill metadata dictionaries.
    """
    skills: List[Dict[str, str]] = []

    for directory in skill_dirs(skills_dir):
        loaded = load_dir(directory)
        names = {skill["name"] for skill in loaded}

        for skill in skills:
            if skill["name"] in names:
                logger.warning(
                    f"Skill '{skill['name']}' in {directory} "
                    f"overrides {skill['path']}"
                )

        skills = [
            skill for skill in skills if skill["name"] not in names
        ] + loaded

    return skills


class DynamicSkillsLoader:
    """
    Dynamic skills loader that uses text similarity to determine which skills
    to load based on the user's task.
    """

    def __init__(
        self,
        skills_dir: Union[str, List[str]],
        similarity_threshold: float = 0.3,
    ):
        """
        Initialize the dynamic skills loader.

        Args:
            skills_dir: Path, or list of paths, to directories containing skill folders with SKILL.md files.
                A later directory's skill replaces an earlier one with the same name.
            similarity_threshold: Minimum similarity score (0-1) for skill loading
        """
        self.skills_dir = skills_dir
        self.similarity_threshold = similarity_threshold
        self.skills_metadata = load_skill_dirs(
            skills_dir, self._load_all_skills_metadata
        )

    def _load_all_skills_metadata(
        self, skills_dir: str
    ) -> List[Dict[str, str]]:
        """
        Load metadata for all available skills from one skills directory.

        Args:
            skills_dir: Path to a directory containing skill folders.

        Returns:
            List of skill metadata dictionaries with name, description, and path
        """
        skills = []

        if not os.path.exists(skills_dir):
            return skills

        for skill_folder in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_folder)

            if not os.path.isdir(skill_path):
                continue

            skill_file = os.path.join(skill_path, "SKILL.md")

            if not os.path.exists(skill_file):
                continue

            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse YAML frontmatter
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = yaml.safe_load(parts[1])
                        skills.append(
                            {
                                "name": frontmatter.get("name", ""),
                                "description": frontmatter.get(
                                    "description", ""
                                ),
                                "path": skill_file,
                                "content": (
                                    parts[2].strip()
                                    if len(parts) > 2
                                    else ""
                                ),
                            }
                        )
            except Exception as e:
                print(f"Error loading skill {skill_folder}: {e}")
                continue

        return skills

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words, removing punctuation and converting to lowercase.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Split into words and filter out empty strings
        return [word for word in text.split() if word]

    def _create_embedding(self, text: str) -> Dict[str, float]:
        """
        Create a simple term frequency embedding for the text.

        Args:
            text: Input text to embed

        Returns:
            Dictionary mapping words to their frequencies
        """
        tokens = self._tokenize_text(text)
        embedding = {}

        for token in tokens:
            embedding[token] = embedding.get(token, 0) + 1

        return embedding

    def _cosine_similarity(
        self,
        embedding1: Dict[str, float],
        embedding2: Dict[str, float],
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First text embedding
            embedding2: Second text embedding

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Get all unique words from both embeddings
        all_words = set(embedding1.keys()) | set(embedding2.keys())

        # Calculate dot product
        dot_product = sum(
            embedding1.get(word, 0) * embedding2.get(word, 0)
            for word in all_words
        )

        # Calculate magnitudes
        magnitude1 = math.sqrt(
            sum(val**2 for val in embedding1.values())
        )
        magnitude2 = math.sqrt(
            sum(val**2 for val in embedding2.values())
        )

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _calculate_task_similarity(
        self, task: str, skill_description: str
    ) -> float:
        """
        Calculate similarity between a task and a skill description.

        Args:
            task: User task description
            skill_description: Skill description from metadata

        Returns:
            Similarity score between 0 and 1
        """
        task_embedding = self._create_embedding(task)
        skill_embedding = self._create_embedding(skill_description)

        return self._cosine_similarity(
            task_embedding, skill_embedding
        )

    def load_relevant_skills(self, task: str) -> List[Dict[str, str]]:
        """
        Load skills that are relevant to the given task based on similarity.

        Args:
            task: User task description

        Returns:
            List of relevant skill metadata dictionaries
        """
        relevant_skills = []

        for skill in self.skills_metadata:
            similarity_score = self._calculate_task_similarity(
                task, skill["description"]
            )

            if similarity_score >= self.similarity_threshold:
                skill_copy = skill.copy()
                skill_copy["similarity_score"] = similarity_score
                relevant_skills.append(skill_copy)

        # Sort by similarity score (highest first)
        relevant_skills.sort(
            key=lambda x: x["similarity_score"], reverse=True
        )

        return relevant_skills
