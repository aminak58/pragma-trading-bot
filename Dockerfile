# Pragma Trading Bot - Production Dockerfile
# Multi-stage build for minimal image size
# Python 3.11 on Debian Bullseye

# ============================================
# Stage 1: Builder - Compile dependencies
# ============================================
FROM python:3.11-slim-bullseye as builder

LABEL maintainer="Pragma Trading Team"
LABEL description="Pragma Trading Bot - HMM Regime Detection + Freqtrade"
LABEL version="0.2.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for TA-Lib and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first (layer caching)
WORKDIR /tmp
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FREQTRADE_USER_DIR="/freqtrade/user_data"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgfortran5 \
    libxml2 \
    libxslt1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib library from builder
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r freqtrade && \
    useradd -r -g freqtrade -u 1000 -d /freqtrade -m freqtrade

# Set working directory
WORKDIR /freqtrade

# Copy application code
COPY --chown=freqtrade:freqtrade . .

# Create necessary directories
RUN mkdir -p user_data/strategies user_data/data logs && \
    chown -R freqtrade:freqtrade /freqtrade

# Switch to non-root user
USER freqtrade

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import freqtrade; print('OK')" || exit 1

# Expose API port (if using REST API)
EXPOSE 8080

# Default command
CMD ["bash"]
