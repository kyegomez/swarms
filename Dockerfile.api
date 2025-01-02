# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from api folder
COPY api/requirements.txt .
RUN pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PORT=8080

# Create app user
RUN useradd -m -s /bin/bash app && \
    mkdir -p /app/logs && \
    chown -R app:app /app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /app/wheels

# Create and activate virtual environment
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir /app/wheels/*

# Copy application code
COPY --chown=app:app ./api ./api

# Switch to app user
USER app

# Create directories for logs
RUN mkdir -p /app/logs

# Required environment variables
ENV SUPABASE_URL="" \
    SUPABASE_SERVICE_KEY="" \
    ENVIRONMENT="production" \
    LOG_LEVEL="info" \
    WORKERS=4 \
    MAX_REQUESTS_PER_MINUTE=60 \
    API_KEY_LENGTH=32

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command
CMD ["sh", "-c", "uvicorn api.api:app --host 0.0.0.0 --port $PORT --workers $WORKERS --log-level $LOG_LEVEL"]