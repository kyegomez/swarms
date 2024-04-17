user_preferences = {
    "subjects": "AI Cognitive Architectures",
    "learning_style": "Visual",
    "challenge_level": "Moderate",
}

# Extracting individual preferences
subjects = user_preferences["subjects"]
learning_style = user_preferences["learning_style"]
challenge_level = user_preferences["challenge_level"]


# Curriculum Design Prompt
CURRICULUM_DESIGN_PROMPT = f"""
Develop a semester-long curriculum tailored to student interests in {subjects}. Focus on incorporating diverse teaching methods suitable for a {learning_style} learning style. 
The curriculum should challenge students at a {challenge_level} level, integrating both theoretical knowledge and practical applications. Provide a detailed structure, including 
weekly topics, key objectives, and essential resources needed.
"""

# Interactive Learning Session Prompt
INTERACTIVE_LEARNING_PROMPT = f"""
Basedon the curriculum, generate an interactive lesson plan for a student of {subjects} that caters to a {learning_style} learning style. Incorporate engaging elements and hands-on activities.
"""

# Sample Lesson Prompt
SAMPLE_TEST_PROMPT = f"""
Create a comprehensive sample test for the first week of the {subjects} curriculum, tailored for a {learning_style} learning style and at a {challenge_level} challenge level.
"""

# Image Generation for Education Prompt
IMAGE_GENERATION_PROMPT = f"""
Develop a stable diffusion prompt for an educational image/visual aid that align with the {subjects} curriculum, specifically designed to enhance understanding for students with a {learning_style} 
learning style. This might include diagrams, infographics, and illustrative representations to simplify complex concepts. Ensure you output a 10/10 descriptive image generation prompt only.
"""
