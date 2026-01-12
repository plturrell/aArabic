# Multi-stage build for DragonflyDB API Client
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

COPY --from=builder /app/target/release/dragonfly-cli /usr/local/bin/dragonfly-cli

ENV DRAGONFLY_URL=redis://dragonfly:6379

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD dragonfly-cli --url ${DRAGONFLY_URL} health || exit 1

ENTRYPOINT ["dragonfly-cli"]
CMD ["--help"]
