# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILDX_VERSION

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set metadata labels
LABEL maintainer="daniel@terragonlabs.com" \
      version="0.1.0" \
      description="SQL Synthesis Agentic Playground" \
      org.opencontainers.image.title="sql-synth-agentic-playground" \
      org.opencontainers.image.description="Natural language to SQL translation system" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/sql-synth-agentic-playground"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/tmp \
    && chown -R appuser:appuser /app

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:$STREAMLIT_SERVER_PORT/_stcore/health || exit 1' > /app/healthcheck.sh \
    && chmod +x /app/healthcheck.sh \
    && chown appuser:appuser /app/healthcheck.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

# Development stage
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install pytest pytest-cov black ruff mypy pre-commit

# Switch back to appuser
USER appuser

# Override command for development
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.runOnSave", "true"]