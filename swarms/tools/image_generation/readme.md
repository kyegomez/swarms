# Image Generation

Contributor: [Sihan Zhao](https://github.com/Sarah816)

# Image Generator Tool

This Markdown document provides an introduction to the `Image Generator` function implemented in the given code. The function is designed to generate an image based on a text description using the Steamship API.

## Function Summary

The `Image Generator` function is a tool that generates an image based on a text description. It utilizes the Steamship API to perform the image generation process.

## Usage

To use the `Image Generator` function, follow the steps below:

1. Set up the required configuration parameters.
2. Ensure that the `STEAMSHIP_API_KEY` environment variable is set.
3. Call the `generate_image` endpoint with a text query to generate an image.
4. The function returns the UUID (Universally Unique Identifier) of the generated image.

## Configuration

The `Image Generator` tool accepts the following configuration parameters:

- **Model Name**: Specifies the model to use for image generation. The default model is DALL-E.
- **Size**: Specifies the desired size of the generated image. The default size is 512x512.
- **Return URLs**: Determines whether to return public URLs for the generated images. By default, this option is disabled.
- **Steamship API Key**: The API key required to authenticate and use the Steamship API. It should be set as the `STEAMSHIP_API_KEY` environment variable.

## Endpoint

The `Image Generator` tool provides the following endpoint:

### `/generate_image`

- **Method**: GET
- **Parameters**:
  - `query`: The detailed text-2-image prompt describing the image to be generated.
- **Returns**:
  - The UUID of the generated image.

## Error Handling

If the `STEAMSHIP_API_KEY` is not provided or is empty, a `RuntimeError` is raised, indicating that the API key should be obtained from the Steamship website and added to the environment variables.

If the tool is unable to generate an image, a `RuntimeError` is raised.
