# Multi-stage build for Memgraph API Client
FROM rust:1.75-slim as builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/aimo-memgraph-cli /usr/local/bin/memgraph-cli

ENV MEMGRAPH_URL=http://memgraph:7687

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD memgraph-cli --url ${MEMGRAPH_URL} health || exit 1

ENTRYPOINT ["memgraph-cli"]
CMD ["--help"]
