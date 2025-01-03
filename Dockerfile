# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set environment variables for Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Set the working directory
WORKDIR /usr/src/app

# Copy the entire project into the container
COPY . .

# Install system dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and pytest
RUN pip install --no-cache-dir poetry pytest

# Configure Poetry and install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Create logs directory with proper permissions
RUN mkdir -p /usr/src/app/logs && chmod -R 777 /usr/src/app/logs

# Add pytest to PATH and verify installation
ENV PATH="/usr/local/bin:/root/.local/bin:$PATH"

# Verify pytest installation
RUN python -m pytest --version

# Set the ENTRYPOINT to use pytest
ENTRYPOINT ["python", "-m", "pytest"]

# Set default command arguments
CMD ["--continue-on-collection-errors", "--tb=short", "--disable-warnings"]