MythGen: A Dynamic New Art Form
Overview

![panel_2](https://github.com/elder-plinius/MythGen/assets/133052465/86bb5784-845b-4db8-a38f-217169ea5201)


MythGen is an Iterative Multimedia Generator that allows users to create their own comic stories based on textual prompts. The system integrates state-of-the-art language and image models to provide a seamless and creative experience.
Features

    Initial Prompting: Kick-start your story with an initial text prompt.
    Artistic Style Suffix: Maintain a consistent artistic style throughout your comic.
    Image Generation: Generate captivating comic panels based on textual captions.
    Caption Generation: Produce engaging captions for each comic panel.
    Interactive Story Building: Select your favorite panels and captions to build your story iteratively.
    Storyboard: View the sequence of your selected panels and their associated captions.
    State Management: Keep track of the current state of your comic generation process.
    User-Friendly Interface: Easy-to-use interface built on Gradio.

Prerequisites
OpenAI API Key

You will need an OpenAI API key to access GPT-3 for generating captions. Follow these steps to obtain one:

    Visit OpenAI's Developer Dashboard.
    Sign up for an API key and follow the verification process.
    Once verified, you will be provided with an API key.

Bing Image Creator Cookie

You should obtain your cookie to run this program. Follow these steps to obtain your cookie:

    Go to Bing Image Creator in your browser and log in to your account.
    Press Ctrl+Shift+J to open developer tools.
    Navigate to the Application section.
    Click on the Cookies section.
    Find the variable _U and copy its value.

How to Use

    Initial Prompt: Start by inputting your initial comic concept.
    Select a Panel: Choose your favorite panel and caption from the generated options.
    Iterate: Use the "Next Part" button to generate the next part of your comic based on your latest selection.
    View Storyboard: See your selected comic panels and captions in a storyboard for a comprehensive view of your comic.
    Finalize: Continue this process until you've created your full comic story.

Installation

bash

pip install -r requirements.txt

Running MythGen

bash

python main.py

This will launch the Gradio interface where you can interact with MythGen.
Dependencies

    Python 3.x
    Gradio
    OpenAI's GPT-3
    DALL-E

Contributing

We welcome contributions! Please read the CONTRIBUTING.md for guidelines on how to contribute to this project.
License

This project is licensed under the MIT License. See LICENSE.md for details.
