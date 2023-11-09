# Search in Wikipedia

Contributor: [Yufei Huang](https://github.com/huangyf530)

# Wikipedia Search Tool

This tool allows you to search for entities, view content, and disambiguate entities on Wikipedia.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Wikipedia Search"
- **description_for_model**: "Plugin for wikipedia"

## Endpoint

The tool provides the following endpoints:

- **/search**: Search for an entity on Wikipedia.
- **/lookup**: Look up a keyword in the current passage.
- **/disambiguation**: Disambiguate an entity name to find other entities with similar names on Wikipedia.

## Function Descriptions

- **search(entity: str) -> str**: This function searches for an entity on Wikipedia and returns the first five sentences if it exists. If not, it returns some related entities to search next. The entity should be a string.
- **lookup(keyword: str) -> str**: This function looks up a keyword in the current passage and returns the next several sentences containing the keyword. The keyword should be a string.
- **disambiguation(entity: str) -> str**: This function disambiguates an entity name to find other entities with similar names on Wikipedia. The entity should be a string.