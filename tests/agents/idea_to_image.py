import pytest
import os
import shutil
from swarms.idea2image import Idea2Image

openai_key = os.getenv("OPENAI_API_KEY")
dalle_cookie = os.getenv("BING_COOKIE")

# Constants for testing
TEST_PROMPT = "Happy fish."
TEST_OUTPUT_FOLDER = "test_images/"
OPENAI_API_KEY = openai_key
DALLE_COOKIE = dalle_cookie


@pytest.fixture(scope="module")
def idea2image_instance():
    # Create an instance of the Idea2Image class
    idea2image = Idea2Image(
        image=TEST_PROMPT,
        openai_api_key=OPENAI_API_KEY,
        cookie=DALLE_COOKIE,
        output_folder=TEST_OUTPUT_FOLDER,
    )
    yield idea2image
    # Clean up the test output folder after testing
    if os.path.exists(TEST_OUTPUT_FOLDER):
        shutil.rmtree(TEST_OUTPUT_FOLDER)


def test_idea2image_instance(idea2image_instance):
    # Check if the instance is created successfully
    assert isinstance(idea2image_instance, Idea2Image)


def test_llm_prompt(idea2image_instance):
    # Test the llm_prompt method
    prompt = idea2image_instance.llm_prompt()
    assert isinstance(prompt, str)


def test_generate_image(idea2image_instance):
    # Test the generate_image method
    idea2image_instance.generate_image()
    # Check if the output folder is created
    assert os.path.exists(TEST_OUTPUT_FOLDER)
    # Check if files are downloaded (assuming DALLE-3 responds with URLs)
    files = os.listdir(TEST_OUTPUT_FOLDER)
    assert len(files) > 0


def test_invalid_openai_api_key():
    # Test with an invalid OpenAI API key
    with pytest.raises(Exception) as exc_info:
        Idea2Image(
            image=TEST_PROMPT,
            openai_api_key="invalid_api_key",
            cookie=DALLE_COOKIE,
            output_folder=TEST_OUTPUT_FOLDER,
        )
    assert "Failed to initialize OpenAIChat" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main()
