# Google Serper Queries

Contributor: [Sihan Zhao](https://github.com/Sarah816)

## Tool Description
The "google_serper" tool allows you to fetch information using the Serper.dev Google Search API. This is a low-cost Google Search API, highly useful when you need to answer questions about current events. The input should be a search query and the output is a JSON object of the query results.

### Tool Specification

- **Name**: google_serper
- **Purpose**: Fetch information using the Serper.dev Google Search API
- **Model Name**: google_serper
- **Model Description**: A tool for querying the Serper.dev Google Search API
- **Logo URL**: [logo](https://your-app-url.com/.well-known/logo.png)
- **Contact Email**: hello@contact.com
- **Legal Information URL**: hello@legal.com

### Core Functionality

1. `search_general`

   This method runs the query through GoogleSearch and parses the result. The result is a JSON object of the query results.

This tool uses a fictional `GoogleSerperAPIWrapper` class to interact with the Google Search API and perform the desired functionality. The actual implementation might need an API wrapper that can interact with the Google Search API.

It's important to note that although the Google Search API wrapper used in this example is fictional, in reality, you would need to find an actual API that can perform Google searches and provide search results. As of my knowledge cutoff in September 2021, Google does not publicly provide its Search API, so you might need to explore alternative methods to retrieve this information while ensuring compliance with Google's terms of service.