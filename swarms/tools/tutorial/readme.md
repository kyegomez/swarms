# Tutorial Service

Contributor: [Xin Cong](https://github.com/congxin95)

# Tutorial Tool

This tool provides a tutorial for a foundation model based on a given objective.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Tutorial"
- **description_for_model**: "Plugin for providing tutorial for a given objective."
- **logo_url**: "https://your-app-url.com/.well-known/logo.png"
- **contact_email**: "xin.cong@outlook.com"
- **legal_info_url**: "hello@legal.com"

## API Key

The tool requires an API key from OpenAI. You can sign up for a free account at https://www.openai.com/, create a new API key, and add it to environment variables.

## Endpoint

The tool provides the following endpoint:

- **/tutorial**: Provide a TODO list as a tutorial for the foundation model based on the given objective. The input should be a text string representing the objective.

## Function Description

- **tutorial(text: str) -> str**: This function provides a TODO list as a tutorial for the foundation model based on the given objective. The text should be a string representing the objective. The function returns a TODO list as a tutorial.