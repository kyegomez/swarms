## `Gemini` Documentation

### Introduction

The Gemini module is a versatile tool for leveraging the power of multimodal AI models to generate content. It allows users to combine textual and image inputs to generate creative and informative outputs. In this documentation, we will explore the Gemini module in detail, covering its purpose, architecture, methods, and usage examples.

#### Purpose

The Gemini module is designed to bridge the gap between text and image data, enabling users to harness the capabilities of multimodal AI models effectively. By providing both a textual task and an image as input, Gemini generates content that aligns with the specified task and incorporates the visual information from the image.

### Installation

Before using Gemini, ensure that you have the required dependencies installed. You can install them using the following commands:

```bash
pip install swarms
pip install google-generativeai
pip install python-dotenv
```

### Class: Gemini

#### Overview

The `Gemini` class is the central component of the Gemini module. It inherits from the `BaseMultiModalModel` class and provides methods to interact with the Gemini AI model. Let's dive into its architecture and functionality.

##### Class Constructor

```python
class Gemini(BaseMultiModalModel):
    def __init__(
        self,
        model_name: str = "gemini-pro",
        gemini_api_key: str = get_gemini_api_key_env,
        *args,
        **kwargs,
    ):
```

| Parameter           | Type    | Description                                                      | Default Value     |
|---------------------|---------|------------------------------------------------------------------|--------------------|
| `model_name`        | str     | The name of the Gemini model.                                    | "gemini-pro"       |
| `gemini_api_key`    | str     | The Gemini API key. If not provided, it is fetched from the environment. | (None)             |

- `model_name`: Specifies the name of the Gemini model to use. By default, it is set to "gemini-pro," but you can specify a different model if needed.

- `gemini_api_key`: This parameter allows you to provide your Gemini API key directly. If not provided, the constructor attempts to fetch it from the environment using the `get_gemini_api_key_env` helper function.

##### Methods

1. **run()**

   ```python
   def run(
       self,
       task: str = None,
       img: str = None,
       *args,
       **kwargs,
   ) -> str:
   ```

   | Parameter     | Type     | Description                                |
   |---------------|----------|--------------------------------------------|
   | `task`        | str      | The textual task for content generation.  |
   | `img`         | str      | The path to the image to be processed.    |
   | `*args`       | Variable | Additional positional arguments.           |
   | `**kwargs`    | Variable | Additional keyword arguments.              |

   - `task`: Specifies the textual task for content generation. It can be a sentence or a phrase that describes the desired content.

   - `img`: Provides the path to the image that will be processed along with the textual task. Gemini combines the visual information from the image with the textual task to generate content.

   - `*args` and `**kwargs`: Allow for additional, flexible arguments that can be passed to the underlying Gemini model. These arguments can vary based on the specific Gemini model being used.

   **Returns**: A string containing the generated content.

   **Examples**:

   ```python
   from swarms.models import Gemini

   # Initialize the Gemini model
   gemini = Gemini()

   # Generate content for a textual task with an image
   generated_content = gemini.run(
       task="Describe this image",
       img="image.jpg",
   )

   # Print the generated content
   print(generated_content)
   ```

   In this example, we initialize the Gemini model, provide a textual task, and specify an image for processing. The `run()` method generates content based on the input and returns the result.

2. **process_img()**

   ```python
   def process_img(
       self,
       img: str = None,
       type: str = "image/png",
       *args,
       **kwargs,
   ):
   ```

   | Parameter     | Type     | Description                                          | Default Value |
   |---------------|----------|------------------------------------------------------|----------------|
   | `img`         | str      | The path to the image to be processed.              | (None)         |
   | `type`        | str      | The MIME type of the image (e.g., "image/png").    | "image/png"    |
   | `*args`       | Variable | Additional positional arguments.                     |
   | `**kwargs`    | Variable | Additional keyword arguments.                        |

   - `img`: Specifies the path to the image that will be processed. It's essential to provide a valid image path for image-based content generation.

   - `type`: Indicates the MIME type of the image. By default, it is set to "image/png," but you can change it based on the image format you're using.

   - `*args` and `**kwargs`: Allow for additional, flexible arguments that can be passed to the underlying Gemini model. These arguments can vary based on the specific Gemini model being used.

   **Raises**: ValueError if any of the following conditions are met:
   - No image is provided.
   - The image type is not specified.
   - The Gemini API key is missing.

   **Examples**:

   ```python
   from swarms.models.gemini import Gemini

   # Initialize the Gemini model
   gemini = Gemini()

   # Process an image
   processed_image = gemini.process_img(
       img="image.jpg",
       type="image/jpeg",
   )

   # Further use the processed image in content generation
   generated_content = gemini.run(
       task="Describe this image",
       img=processed_image,
   )

   # Print the generated content
   print(generated_content)
   ```

   In this example, we demonstrate how to process an image using the `process_img()` method and then use the processed image in content generation.

#### Additional Information

- Gemini is designed to work seamlessly with various multimodal AI models, making it a powerful tool for content generation tasks.

- The module uses the `google.generativeai` package to access the underlying AI models. Ensure that you have this package installed to leverage the full capabilities of Gemini.

- It's essential to provide a valid Gemini API key for authentication. You can either pass it directly during initialization or store it in the environment variable "GEMINI_API_KEY."

- Gemini's flexibility allows you to experiment with different Gemini models and tailor the content generation process to your specific needs.

- Keep in mind that Gemini is designed to handle both textual and image inputs, making it a valuable asset for various applications, including natural language processing and computer vision tasks.

- If you encounter any issues or have specific requirements, refer to the Gemini documentation for more details and advanced usage.

### References and Resources

- [Gemini GitHub Repository](https://github.com/swarms/gemini): Explore the Gemini repository for additional information, updates, and examples.

- [Google GenerativeAI Documentation](https://docs.google.com/document/d/1WZSBw6GsOhOCYm0ArydD_9uy6nPPA1KFIbKPhjj43hA): Dive deeper into the capabilities of the Google GenerativeAI package used by Gemini.

- [Gemini API Documentation](https://gemini-api-docs.example.com): Access the official documentation for the Gemini API to explore advanced features and integrations.

## Conclusion

In this comprehensive documentation, we've explored the Gemini module, its purpose, architecture, methods, and usage examples. Gemini empowers developers to generate content by combining textual tasks and images, making it a valuable asset for multimodal AI applications. Whether you're working on natural language processing or computer vision projects, Gemini can help you achieve impressive results.