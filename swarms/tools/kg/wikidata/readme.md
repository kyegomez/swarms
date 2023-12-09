# Wikidata Query

Contributor: [Yuzhang Zhu](https://github.com/Zhu-Yuzhang)

# Search in Wikidata Tool

This Markdown document provides an introduction to the `Search in Wikidata` function implemented in the given code. The function allows users to search for factual information in Wikidata by querying entities, relations, and performing SPARQL queries.

## Function Summary

The `Search in Wikidata` function is a tool that enables users to search for factual information in Wikidata. It provides functionality for finding entities, retrieving entity details, searching for relations, and performing SPARQL queries.

## Usage

To use the `Search in Wikidata` function, follow the steps below:

1. Call the desired endpoint with appropriate parameters to perform the specific search or query operation.
2. The function returns a markdown-formatted table containing the results.

## Endpoints

The `Search in Wikidata` tool provides the following endpoints:

### `/find_entity`

- **Method**: GET
- **Parameters**:
  - `input`: The input entity to find relations and properties for.

### `/find_entity_by_tail`

- **Method**: GET
- **Parameters**:
  - `input`: The input tail to find head entities and relations for.

### `/get_entity_id`

- **Method**: GET
- **Parameters**:
  - `input`: The surface form to search for entities.

### `/get_relation_id`

- **Method**: GET
- **Parameters**:
  - `input`: The surface form to search for relations.

### `/search_by_code`

- **Method**: GET
- **Parameters**:
  - `query`: The SPARQL query to perform.

## Error Handling

If any invalid options or exceptions occur during the execution of the function, appropriate error messages are printed.
