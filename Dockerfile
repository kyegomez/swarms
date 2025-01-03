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

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and dependencies
RUN pip install --no-cache-dir poetry pytest

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Create logs directory with proper permissions
RUN mkdir -p /usr/src/app/logs && chmod -R 777 /usr/src/app/logs

# Add pytest to PATH and verify installation
ENV PATH="/usr/local/bin:/root/.local/bin:$PATH"
RUN python -m pytest --version

# Set the default command
ENTRYPOINT ["pytest"]
CMD ["/usr/src/app/tests", "--continue-on-collection-errors", "--tb=short", "--disable-warnings"]