<<<<<<< HEAD
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

# Install Poetry
RUN pip install poetry

# Configure Poetry to avoid virtual environments and install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Install additional dependencies outside Poetry (e.g., swarms, pytest)
RUN pip install swarms pytest

# Ensure pytest is installed and available
RUN pytest --version

# Ensure the logs directory has correct permissions (in case of permission issues with mounted volumes)
RUN mkdir -p /usr/src/app/logs && chmod -R 777 /usr/src/app/logs

# Ensure that the PATH includes the directory where pytest is installed
ENV PATH="/usr/local/bin:$PATH"

# Set the working directory to the tests directory inside the container
WORKDIR /usr/src/app/tests

# Default command to run tests located in the /tests directory
CMD pytest /usr/src/app/tests --continue-on-collection-errors --tb=short --disable-warnings | tee /usr/src/app/logs/test_logs.txt
=======
# Use Python 3.11 slim-bullseye for smaller base image
FROM python:3.11-slim-bullseye AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory
WORKDIR /build

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install swarms packages
RUN pip install --no-cache-dir swarm-models swarms

# Production stage
FROM python:3.11-slim-bullseye

# Set secure environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKSPACE_DIR="agent_workspace" \
    PATH="/app:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    USER=swarms

# Create non-root user
RUN useradd -m -s /bin/bash -U $USER && \
    mkdir -p /app && \
    chown -R $USER:$USER /app

# Set working directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application with correct permissions
COPY --chown=$USER:$USER . .

# Switch to non-root user
USER $USER

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import swarms; print('Health check passed')" || exit 1
>>>>>>> upstream/master
