# Google Places Queries

Contributor: [Sihan Zhao](https://github.com/Sarah816)

## Tool Description
The "Google Places" tool allows you to fetch information about places and locations by querying the Google Places API. This tool can be especially useful when you need to validate or discover addresses from ambiguous text.

### Tool Specifications

- **Name**: Google Places
- **Purpose**: Look up for information about places and locations
- **Name for Model**: google_places
- **Model Description**: A tool that query the Google Places API. Useful for when you need to validate or discover addressed from ambiguous text. Input should be a search query.
- **Contact Email**: hello@contact.com
- **Legal Information URL**: hello@legal.com

### Core Functionality

1. `search_places`

   This method accepts a string argument that contains a search query. It queries the Google Places API with the given input and returns information about the corresponding places and locations.

The tool leverages a fictional wrapper class called `GooglePlacesAPIWrapper` to interact with the Google Places API and execute the required functionalities.

**note**: The GooglePlacesAPIWrapper is a placeholder here and in a real-world scenario, this should be replaced with a properly implemented class that can fetch data from the Google Places API. The Google Places API itself requires an API key for access, which is not provided in this code.