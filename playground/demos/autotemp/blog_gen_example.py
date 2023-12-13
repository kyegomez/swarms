import os
from blog_gen import BlogGen


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set."
        )

    blog_topic = input("Enter the topic for the blog generation: ")

    blog_generator = BlogGen(api_key, blog_topic)
    blog_generator.TOPIC_SELECTION_SYSTEM_PROMPT = (
        blog_generator.TOPIC_SELECTION_SYSTEM_PROMPT.replace(
            "{{BLOG_TOPIC}}", blog_topic
        )
    )

    blog_generator.run_workflow()


if __name__ == "__main__":
    main()
