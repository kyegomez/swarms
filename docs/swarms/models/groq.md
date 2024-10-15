# Groq API Key Setup Documentation


This documentation provides instructions on how to obtain your Groq API key and set it up in a `.env` file for use in your project.

## Step 1: Obtain Your Groq API Key

1. **Sign Up / Log In**: 
   - Visit the [Groq website](https://www.groq.com) and sign up for an account if you don't have one. If you already have an account, log in.

2. **Access API Keys**:
   - Once logged in, navigate to the API section of your account dashboard. This is usually found under "Settings" or "API Management".

3. **Generate API Key**:
   - If you do not have an API key, look for an option to generate a new key. Follow the prompts to create your API key. Make sure to copy it to your clipboard.

## Step 2: Create a `.env` File

1. **Create the File**:
   - In the root directory of your project, create a new file named `.env`.

2. **Add Your API Key**:
   - Open the `.env` file in a text editor and add the following line, replacing `your_groq_api_key_here` with the API key you copied earlier:

   ```plaintext
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Save the File**:
   - Save the changes to the `.env` file.



## Full Example
```python
import os
from swarms import Agent, SequentialWorkflow
from swarm_models import OpenAIChat

# model = Anthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
company = "TGSC"
# Get the OpenAI API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Model
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)

model.run("What are the best metrics to track and understand risk in private equity")
```

## Important Notes

- **Keep Your API Key Secure**: Do not share your API key publicly or commit it to version control systems like Git. Use the `.gitignore` file to exclude the `.env` file from being tracked.
- **Environment Variables**: Make sure to install any necessary libraries (like `python-dotenv`) to load environment variables from the `.env` file if your project requires it.


## Conclusion

You are now ready to use the Groq API in your project! If you encounter any issues, refer to the Groq documentation or support for further assistance.
