# TESTING
# -==================
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to make Python output unbuffered and disable the PIP cache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK on
ENV PIP_DEFAULT_TIMEOUT 100

# Set environment variables for OpenAI API key and workspace directory
ENV OPENAI_API_KEY your_api_key_here
ENV WORKSPACE_DIR /path/to/your/workspace

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Poetry
RUN pip install poetry

# Disable virtualenv creation by poetry and install dependencies
RUN poetry config virtualenvs.create false

# Install the 'swarms' package and any additional packages
RUN pip install swarms swarm-models swarms-memory pytest

# Run all tests in the tests directory
CMD python3 -m unittest discover -s tests