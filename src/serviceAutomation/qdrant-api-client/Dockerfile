# Multi-stage build for Qdrant API Client
FROM rust:1.75-slim as builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/qdrant-cli /usr/local/bin/qdrant-cli

# Set environment variables
ENV QDRANT_URL=http://qdrant:6333

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD qdrant-cli --url ${QDRANT_URL} health || exit 1

ENTRYPOINT ["qdrant-cli"]
CMD ["--help"]
