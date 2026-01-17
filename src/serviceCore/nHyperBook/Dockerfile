# HyperShimmy Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM zigimg/zig:0.12.0 AS builder

WORKDIR /build

# Copy source files
COPY build.zig .
COPY server/ server/
COPY io/ io/
COPY mojo/ mojo/
COPY webapp/ webapp/
COPY tests/ tests/

# Build release binary
RUN zig build -Doptimize=ReleaseFast

# Stage 2: Runtime
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 hypershimmy

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/zig-out/bin/hypershimmy /app/hypershimmy
COPY --from=builder /build/webapp /app/webapp

# Set ownership
RUN chown -R hypershimmy:hypershimmy /app

# Switch to non-root user
USER hypershimmy

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PORT=8080 \
    HOST=0.0.0.0 \
    LOG_LEVEL=INFO

# Run application
CMD ["./hypershimmy"]
