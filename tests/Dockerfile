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
RUN poetry config vir# Use Python 3.11 slim-bullseye for smaller base image
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
    CMD python -c "import swarms; print('Health check passed')" || exit 1tualenvs.create false

# Install the 'swarms' package if it's not included in the poetry.lock
RUN pip install swarms

# Assuming tests require pytest to run
RUN pip install pytest

# Run pytest on all tests in the tests directory
CMD pytest
