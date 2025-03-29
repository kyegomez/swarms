# Use a lightweight Python image
FROM python:3.11-slim-bullseye

# Environment config for speed and safety
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    USER=swarms

# Set working directory
WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the swarms package
RUN pip install --upgrade pip && pip install -U swarms

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
