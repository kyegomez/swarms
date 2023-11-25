# Douban Film Search

Contributor: [Jing Yi](https://github.com/yijing16)

## Tool Description
The "Film Search Plugin" is a robust tool that allows you to access up-to-date film information, filter through this information, and retrieve detailed descriptions of specific films. It utilizes a fictional API called "DoubanAPI" to pull information on films.

### Tool Specifications

- **Name**: Film Search Plugin
- **Purpose**: Search for up-to-date film information.
- **Name for Model**: Film Search
- **Model Description**: Plugin for search for up-to-date film information.
- **Logo URL**: [logo](https://your-app-url.com/.well-known/logo.png)
- **Contact Email**: hello@contact.com
- **Legal Information URL**: hello@legal.com

### Core Functionality

1. `coming_out_filter`

    This method accepts a string argument that contains details about the desired region, category, number of films, and a flag indicating if sorting is needed based on the number of people looking forward to the movie. It filters through the upcoming films based on these details and provides the information.

2. `now_playing_out_filter`

    This method accepts a string argument that contains details about the desired region, the number of films, and a flag indicating if sorting is needed based on the film score. It filters through the currently playing films based on these details and provides the information.

3. `print_detail`

    This method accepts a string argument that contains the name of a specific film. It provides detailed information about the film, including the genre, director, actors, and a brief synopsis of the film's plot.

All methods use the `DoubanAPI` to retrieve and filter information on films.

**Note**: This tool's functionality is hypothetical and relies on the existence and proper functioning of a fictional API, DoubanAPI, which is not included in the provided code. In a real-world application, replace DoubanAPI with a functional API that can retrieve film data.