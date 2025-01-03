# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set environment variables for Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Set the working directory to the root of the project (relative to pyproject.toml)
WORKDIR /usr/src/app

# Copy the entire project into the container
COPY . .

# Install Poetry and necessary dependencies
RUN pip install poetry

# Configure Poetry to avoid virtual environments and install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Install pytest
RUN pip install pytest

# Ensure pytest is installed and available
RUN pytest --version || echo "pytest not found"

# Ensure the logs directory has correct permissions
RUN mkdir -p /usr/src/app/logs && chmod -R 777 /usr/src/app/logs

# Ensure that the PATH includes the directory where pytest is installed
ENV PATH="/usr/local/bin:$PATH"

# List files to verify pytest and tests exist
RUN ls -l /usr/src/app

# Set the working directory to the tests directory inside the container
WORKDIR /usr/src/app/tests

# Default command to run tests located in the /tests directory
CMD pytest /usr/src/app/tests --continue-on-collection-errors --tb=short --disable-warnings | tee /usr/src/app/logs/test_logs.txt
