# Use Python 3.11 slim-bullseye for a smaller base image
FROM python:3.11-slim-bullseye

# Set environment variables for Python and pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WORKSPACE_DIR="agent_workspace" \
    PATH="/app:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    USER=swarms

# Set the working directory
WORKDIR /app

# Install essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir swarm-models swarms && \
    pip install --no-cache-dir transformers torch litellm tiktoken openai pandas numpy pypdf

# Create a non-root user and set correct permissions for the application directory
RUN useradd -m -s /bin/bash -U $USER && \
    mkdir -p /app && \
    chown -R $USER:$USER /app

# Copy application files into the image with proper ownership
COPY --chown=$USER:$USER . .

# Switch to the non-root user
USER $USER

# Health check to ensure the container is running properly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import swarms; print('Health check passed')" || exit 1
