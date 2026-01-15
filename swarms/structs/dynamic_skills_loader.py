"""
Dynamic Skills Loader for Agent

This module provides intelligent skill loading based on task similarity.
It tokenizes and embeds both user tasks and skill descriptions, then uses
cosine similarity to determine which skills should be loaded into the agent.
"""

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import yaml


class DynamicSkillsLoader:
    """
    Dynamic skills loader that uses text similarity to determine which skills
    to load based on the user's task.
    """

    def __init__(
        self, skills_dir: str, similarity_threshold: float = 0.3
    ):
        """
        Initialize the dynamic skills loader.

        Args:
            skills_dir: Path to directory containing skill folders with SKILL.md files
            similarity_threshold: Minimum similarity score (0-1) for skill loading
        """
        self.skills_dir = skills_dir
        self.similarity_threshold = similarity_threshold
        self.skills_metadata = self._load_all_skills_metadata()

    def _load_all_skills_metadata(self) -> List[Dict[str, str]]:
        """
        Load metadata for all available skills from the skills directory.

        Returns:
            List of skill metadata dictionaries with name, description, and path
        """
        skills = []

        if not os.path.exists(self.skills_dir):
            return skills

        for skill_folder in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, skill_folder)

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

    def get_skill_names(self, task: str) -> List[str]:
        """
        Get names of relevant skills for a task.

        Args:
            task: User task description

        Returns:
            List of skill names
        """
        relevant_skills = self.load_relevant_skills(task)
        return [skill["name"] for skill in relevant_skills]

    def load_full_skill_content(
        self, skill_name: str
    ) -> Optional[str]:
        """
        Load the full content of a specific skill.

        Args:
            skill_name: Name of the skill to load

        Returns:
            Full skill content or None if not found
        """
        for skill in self.skills_metadata:
            if skill["name"] == skill_name:
                return skill["content"]
        return None

    def get_similarity_scores(
        self, task: str
    ) -> List[Tuple[str, float]]:
        """
        Get similarity scores for all skills relative to a task.

        Args:
            task: User task description

        Returns:
            List of tuples (skill_name, similarity_score)
        """
        scores = []
        for skill in self.skills_metadata:
            similarity_score = self._calculate_task_similarity(
                task, skill["description"]
            )
            scores.append((skill["name"], similarity_score))

        # Sort by similarity score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def create_dynamic_skills_loader(
    skills_dir: str, similarity_threshold: float = 0.3
) -> DynamicSkillsLoader:
    """
    Factory function to create a dynamic skills loader.

    Args:
        skills_dir: Path to skills directory
        similarity_threshold: Minimum similarity threshold

    Returns:
        Configured DynamicSkillsLoader instance
    """
    return DynamicSkillsLoader(skills_dir, similarity_threshold)
