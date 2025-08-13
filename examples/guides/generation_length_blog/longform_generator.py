import time
from typing import Dict, List

from swarms import Agent
from swarms.utils.litellm_tokenizer import count_tokens


class LongFormGenerator:
    """
    A class for generating long-form content using the swarms Agent framework.

    This class provides methods for creating comprehensive, detailed content
    with support for continuation and sectioned generation.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the LongFormGenerator with specified model.

        Args:
            model (str): The model to use for content generation
        """
        self.model = model

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text (str): The text to estimate tokens for

        Returns:
            int: Estimated token count
        """
        return count_tokens(text=text, model=self.model)

    def create_expansion_prompt(
        self, topic: str, requirements: Dict
    ) -> str:
        """
        Create optimized prompt for long-form content.

        Args:
            topic (str): The main topic to generate content about
            requirements (Dict): Requirements for content generation

        Returns:
            str: Formatted prompt for content generation
        """
        structure_requirements = []
        if "sections" in requirements:
            for i, section in enumerate(requirements["sections"]):
                structure_requirements.append(
                    f"{i+1}. {section['title']} - {section.get('description', 'Provide comprehensive analysis')}"
                )

        length_guidance = (
            f"Target length: {requirements.get('min_words', 2000)}-{requirements.get('max_words', 4000)} words"
            if "min_words" in requirements
            else ""
        )

        prompt = f"""Create a comprehensive, detailed analysis of: {topic}
REQUIREMENTS:
- This is a professional-level document requiring thorough treatment
- Each section must be substantive with detailed explanations
- Include specific examples, case studies, and technical details where relevant
- Provide multiple perspectives and comprehensive coverage
- {length_guidance}
STRUCTURE:
{chr(10).join(structure_requirements)}
QUALITY STANDARDS:
- Demonstrate deep expertise and understanding
- Include relevant technical specifications and details
- Provide actionable insights and practical applications  
- Use professional language appropriate for expert audience
- Ensure logical flow and comprehensive coverage of all aspects
Begin your comprehensive analysis:"""

        return prompt

    def generate_with_continuation(
        self, topic: str, requirements: Dict, max_attempts: int = 3
    ) -> str:
        """
        Generate long-form content with continuation if needed.

        Args:
            topic (str): The main topic to generate content about
            requirements (Dict): Requirements for content generation
            max_attempts (int): Maximum number of continuation attempts

        Returns:
            str: Generated long-form content
        """
        initial_prompt = self.create_expansion_prompt(
            topic, requirements
        )

        # Create agent for initial generation
        agent = Agent(
            name="LongForm Content Generator",
            system_prompt=initial_prompt,
            model=self.model,
            max_loops=1,
            temperature=0.7,
            max_tokens=4000,
        )

        # Generate initial response
        content = agent.run(topic)
        target_words = requirements.get("min_words", 2000)

        # Check if continuation is needed
        word_count = len(content.split())
        continuation_count = 0

        while (
            word_count < target_words
            and continuation_count < max_attempts
        ):
            continuation_prompt = f"""Continue and expand the previous analysis. The current response is {word_count} words, but we need approximately {target_words} words total for comprehensive coverage.
Please continue with additional detailed analysis, examples, and insights. Focus on areas that could benefit from deeper exploration or additional perspectives. Maintain the same professional tone and analytical depth.
Continue the analysis:"""

            # Create continuation agent
            continuation_agent = Agent(
                name="Content Continuation Agent",
                system_prompt=continuation_prompt,
                model=self.model,
                max_loops=1,
                temperature=0.7,
                max_tokens=4000,
            )

            # Generate continuation
            continuation_content = continuation_agent.run(
                f"Continue the analysis on: {topic}"
            )
            content += "\n\n" + continuation_content
            word_count = len(content.split())
            continuation_count += 1

            # Rate limiting
            time.sleep(1)

        return content

    def generate_sectioned_content(
        self,
        topic: str,
        sections: List[Dict],
        combine_sections: bool = True,
    ) -> Dict:
        """
        Generate content section by section for maximum length.

        Args:
            topic (str): The main topic to generate content about
            sections (List[Dict]): List of section definitions
            combine_sections (bool): Whether to combine all sections into one document

        Returns:
            Dict: Dictionary containing individual sections and optionally combined content
        """
        results = {}
        combined_content = ""

        for section in sections:
            section_prompt = f"""Write a comprehensive, detailed section on: {section['title']}
Context: This is part of a larger analysis on {topic}
Requirements for this section:
- Provide {section.get('target_words', 500)}-{section.get('max_words', 800)} words of detailed content
- {section.get('description', 'Provide thorough analysis with examples and insights')}
- Include specific examples, technical details, and practical applications
- Use professional language suitable for expert audience
- Ensure comprehensive coverage of all relevant aspects
Write the complete section:"""

            # Create agent for this section
            section_agent = Agent(
                name=f"Section Generator - {section['title']}",
                system_prompt=section_prompt,
                model=self.model,
                max_loops=1,
                temperature=0.7,
                max_tokens=3000,
            )

            # Generate section content
            section_content = section_agent.run(
                f"Generate section: {section['title']} for topic: {topic}"
            )
            results[section["title"]] = section_content

            if combine_sections:
                combined_content += (
                    f"\n\n## {section['title']}\n\n{section_content}"
                )

            # Rate limiting between sections
            time.sleep(1)

        if combine_sections:
            results["combined"] = combined_content.strip()

        return results


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = LongFormGenerator()

    # Example topic and requirements
    topic = "Artificial Intelligence in Healthcare"
    requirements = {
        "min_words": 2500,
        "max_words": 4000,
        "sections": [
            {
                "title": "Current Applications",
                "description": "Analyze current AI applications in healthcare",
                "target_words": 600,
                "max_words": 800,
            },
            {
                "title": "Future Prospects",
                "description": "Discuss future developments and potential",
                "target_words": 500,
                "max_words": 700,
            },
        ],
    }

    # Generate comprehensive content
    content = generator.generate_with_continuation(
        topic, requirements
    )
    print("Generated Content:")
    print(content)
    print(f"\nWord count: {len(content.split())}")

    # Generate sectioned content
    sections = [
        {
            "title": "AI in Medical Imaging",
            "description": "Comprehensive analysis of AI applications in medical imaging",
            "target_words": 500,
            "max_words": 700,
        },
        {
            "title": "AI in Drug Discovery",
            "description": "Detailed examination of AI in pharmaceutical research",
            "target_words": 600,
            "max_words": 800,
        },
    ]

    sectioned_results = generator.generate_sectioned_content(
        topic, sections
    )
    print("\nSectioned Content:")
    for section_title, section_content in sectioned_results.items():
        if section_title != "combined":
            print(f"\n--- {section_title} ---")
            print(section_content[:200] + "...")
