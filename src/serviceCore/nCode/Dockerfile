# nCode - Code Intelligence Platform
# Multi-stage Dockerfile for production deployment
#
# Build with: docker build -t ncode:latest .
# Run with: docker run -p 18003:18003 ncode:latest

# ============================================================================
# Stage 1: Build Zig server
# ============================================================================
FROM alpine:3.19 AS zig-builder

# Install build dependencies
RUN apk add --no-cache \
    curl \
    tar \
    xz \
    gcc \
    musl-dev \
    git

# Install Zig compiler
ARG ZIG_VERSION=0.11.0
RUN curl -L https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz \
    | tar -xJ -C /usr/local && \
    ln -s /usr/local/zig-linux-x86_64-${ZIG_VERSION}/zig /usr/local/bin/zig

# Set working directory
WORKDIR /build

# Copy Zig source files
COPY build.zig .
COPY zig_scip_writer.zig .
COPY scip_reader.zig .
COPY treesitter_indexer.zig .
COPY server/ ./server/

# Build nCode server
RUN zig build -Doptimize=ReleaseFast

# ============================================================================
# Stage 2: Python environment for loaders
# ============================================================================
FROM python:3.11-slim AS python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy Python loader modules
COPY loaders/ ./loaders/
COPY scripts/ ./scripts/

# ============================================================================
# Stage 3: Final runtime image
# ============================================================================
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ncode && useradd -r -g ncode ncode

# Set working directory
WORKDIR /app

# Copy Zig binary from builder
COPY --from=zig-builder /build/zig-out/bin/ ./bin/

# Copy Python dependencies from builder
COPY --from=python-builder /root/.local /home/ncode/.local

# Copy Python modules
COPY --from=python-builder /build/loaders ./loaders
COPY --from=python-builder /build/scripts ./scripts

# Copy additional files
COPY docs/ ./docs/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Create directories for data and logs
RUN mkdir -p /app/indexes /app/logs /app/backups && \
    chown -R ncode:ncode /app

# Set Python path
ENV PATH=/home/ncode/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Switch to non-root user
USER ncode

# Expose nCode server port
EXPOSE 18003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:18003/health || exit 1

# Default command: start nCode server
CMD ["/app/bin/ncode-server"]
