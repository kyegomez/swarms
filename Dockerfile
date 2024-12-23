# Use Python 3.11 instead of 3.13
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKSPACE_DIR="agent_workspace" \
    OPENAI_API_KEY="your_swarm_api_key_here"

# Set the working directory
WORKDIR /usr/src/swarms

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install swarms package
RUN pip3 install -U swarm-models
RUN pip3 install -U swarms

# Copy the application
COPY . .