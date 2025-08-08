# Multi-stage build for optimized Docker image
FROM python:3.11-slim-bullseye as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV for faster package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create a virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the swarms package using UV
RUN uv pip install --system -U swarms

# Final stage
FROM python:3.11-slim-bullseye

# Environment config for speed and safety
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    USER=swarms

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Add non-root user
RUN useradd -m -s /bin/bash -U $USER && \
    chown -R $USER:$USER /app

# Copy application code
COPY --chown=$USER:$USER . .

# Switch to non-root
USER $USER

# Optional health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import swarms; print('Health check passed')" || exit 1
