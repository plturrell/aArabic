# Build stage
FROM alpine:3.19 as builder

# Install Zig and build dependencies
RUN apk add --no-cache \
    zig \
    git \
    curl \
    libstdc++

WORKDIR /app

# Copy source code
COPY . .

# Build the server in release mode
RUN zig build -Doptimize=ReleaseFast

# Runtime stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    libstdc++ \
    ca-certificates \
    tzdata

# Create non-root user for security
RUN addgroup -S nworkflow && adduser -S nworkflow -G nworkflow

# Copy binary from builder
COPY --from=builder /app/zig-out/bin/nworkflow_server /usr/local/bin/

# Copy webapp assets
COPY --from=builder /app/webapp /app/webapp

# Set working directory
WORKDIR /app

# Change ownership
RUN chown -R nworkflow:nworkflow /app

# Switch to non-root user
USER nworkflow

# Expose the server port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8090/health || exit 1

# Run the server
CMD ["nworkflow_server"]

