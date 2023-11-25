# TESTING
# -==================
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to make Python output unbuffered and disable the PIP cache
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK on
ENV PIP_DEFAULT_TIMEOUT 100

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Poetry
RUN pip install poetry

# Disable virtualenv creation by poetry and install dependencies
RUN poetry config virtualenvs.create false

# Install the 'swarms' package if it's not included in the poetry.lock
RUN pip install swarms

# Assuming tests require pytest to run
RUN pip install pytest

# Run pytest on all tests in the tests directory
CMD pytest
