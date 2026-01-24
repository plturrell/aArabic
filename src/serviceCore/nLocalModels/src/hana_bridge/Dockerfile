# HANA Cloud SQL Bridge - Production Dockerfile
# Multi-stage build for minimal image size

# ============================================================================
# Build Stage
# ============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install build dependencies for native modules
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production

# ============================================================================
# Production Stage
# ============================================================================
FROM node:20-alpine

WORKDIR /app

# Install tini for proper signal handling
RUN apk add --no-cache tini

# Create non-root user
RUN addgroup -g 1001 -S hana && \
    adduser -u 1001 -S hana -G hana

# Copy node_modules from builder
COPY --from=builder /app/node_modules ./node_modules

# Copy application files
COPY server.prod.js ./server.js
COPY package.json ./

# Set ownership
RUN chown -R hana:hana /app

# Switch to non-root user
USER hana

# Environment defaults
ENV NODE_ENV=production \
    BRIDGE_PORT=3001 \
    BRIDGE_HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    POOL_MIN=2 \
    POOL_MAX=10

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:3001/health || exit 1

# Expose port
EXPOSE 3001

# Use tini as init
ENTRYPOINT ["/sbin/tini", "--"]

# Start application
CMD ["node", "server.js"]
