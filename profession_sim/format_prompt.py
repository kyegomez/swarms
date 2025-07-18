#!/usr/bin/env python3
"""
Script to format prompt.txt into proper markdown format.
Converts \n characters to actual line breaks and improves formatting.
"""


def format_prompt(
    input_file="prompt.txt", output_file="prompt_formatted.md"
):
    """
    Read the prompt file and format it properly as markdown.

    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
    """
    try:
        # Read the original file
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace \n with actual newlines
        formatted_content = content.replace("\\n", "\n")

        # Additional formatting improvements
        # Fix spacing around headers
        formatted_content = formatted_content.replace(
            "\n**", "\n\n**"
        )
        formatted_content = formatted_content.replace(
            "**\n", "**\n\n"
        )

        # Fix spacing around list items
        formatted_content = formatted_content.replace(
            "\n  -", "\n\n  -"
        )

        # Fix spacing around sections
        formatted_content = formatted_content.replace(
            "\n---\n", "\n\n---\n\n"
        )

        # Clean up excessive newlines (more than 3 in a row)
        import re

        formatted_content = re.sub(
            r"\n{4,}", "\n\n\n", formatted_content
        )

        # Write the formatted content
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        print("✅ Successfully formatted prompt!")
        print(f"📄 Input file: {input_file}")
        print(f"📝 Output file: {output_file}")

        # Show some stats
        original_lines = content.count("\\n") + 1
        new_lines = formatted_content.count("\n") + 1
        print(f"📊 Lines: {original_lines} → {new_lines}")

    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{input_file}'")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    format_prompt()
