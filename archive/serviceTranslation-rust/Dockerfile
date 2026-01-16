# Multi-stage Dockerfile for Arabic Translation Service
# Pure Rust implementation - no Python dependencies!

# Stage 1: Build the Rust binary
FROM rust:1.75-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY examples ./examples

# Build release binary
RUN cargo build --release --bin serve

# Stage 2: Runtime image
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /build/target/release/serve /usr/local/bin/serve

# Copy model files (or mount as volume)
# COPY models/ /app/models/

# Create non-root user
RUN useradd -m -u 1000 translator && \
    chown -R translator:translator /app

USER translator

# Expose port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Run the service
ENTRYPOINT ["serve"]
CMD ["--port", "8090"]

# Labels
LABEL org.opencontainers.image.title="Arabic Translation Service"
LABEL org.opencontainers.image.description="Pure Rust M2M100 translation service"
LABEL org.opencontainers.image.vendor="AI Nucleus"
LABEL org.opencontainers.image.version="1.0.0"
