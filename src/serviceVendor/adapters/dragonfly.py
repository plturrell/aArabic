"""
DragonflyDB Adapter
Integration layer for DragonflyDB (Redis-compatible in-memory data store).
Provides caching, session management, and pub/sub functionality.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis


@dataclass
class DragonflyHealthStatus:
    """Health status of the DragonflyDB service."""
    healthy: bool
    status: str
    version: Optional[str] = None
    memory_used: Optional[int] = None
    connected_clients: Optional[int] = None
    error: Optional[str] = None


class DragonflyAdapter:
    """
    Adapter for interacting with DragonflyDB.

    Provides methods for:
    - Key-value operations
    - Caching with TTL
    - JSON document storage
    - Pub/Sub messaging
    - Health monitoring
    """

    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True
    ):
        self.url = url or os.getenv("DRAGONFLY_URL", "redis://localhost:6379")
        self.host = host or os.getenv("DRAGONFLY_HOST", "localhost")
        self.port = port
        self.db = db
        self.password = password or os.getenv("DRAGONFLY_PASSWORD")
        self.decode_responses = decode_responses
        self._client: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            if self.url:
                self._client = aioredis.from_url(
                    self.url,
                    decode_responses=self.decode_responses
                )
            else:
                self._client = aioredis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=self.decode_responses
                )
        return self._client

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> DragonflyHealthStatus:
        """Check the health of DragonflyDB."""
        try:
            client = await self._get_client()
            info = await client.info()
            return DragonflyHealthStatus(
                healthy=True,
                status="healthy",
                version=info.get("redis_version"),
                memory_used=info.get("used_memory"),
                connected_clients=info.get("connected_clients")
            )
        except Exception as e:
            return DragonflyHealthStatus(
                healthy=False,
                status="unavailable",
                error=str(e)
            )

    # Key-Value Operations
    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        client = await self._get_client()
        return await client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value with optional TTL in seconds."""
        client = await self._get_client()
        return await client.set(key, value, ex=ttl)

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        client = await self._get_client()
        return await client.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        client = await self._get_client()
        return await client.exists(*keys)

    # JSON Operations
    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Store a JSON-serializable value."""
        return await self.set(key, json.dumps(value), ttl=ttl)

    async def get_json(self, key: str) -> Optional[Any]:
        """Retrieve and parse a JSON value."""
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None

    # Hash Operations
    async def hset(self, name: str, key: str, value: str) -> int:
        """Set a hash field."""
        client = await self._get_client()
        return await client.hset(name, key, value)

    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get a hash field."""
        client = await self._get_client()
        return await client.hget(name, key)

    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        client = await self._get_client()
        return await client.hgetall(name)

    # List Operations
    async def lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list."""
        client = await self._get_client()
        return await client.lpush(key, *values)

    async def rpush(self, key: str, *values: str) -> int:
        """Push values to the right of a list."""
        client = await self._get_client()
        return await client.rpush(key, *values)

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get a range of list elements."""
        client = await self._get_client()
        return await client.lrange(key, start, end)

    # Cache Helpers
    async def cache_get_or_set(
        self,
        key: str,
        factory,
        ttl: int = 3600
    ) -> Any:
        """Get from cache or compute and store."""
        value = await self.get_json(key)
        if value is not None:
            return value

        value = await factory() if callable(factory) else factory
        await self.set_json(key, value, ttl=ttl)
        return value

    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        client = await self._get_client()
        keys = []
        async for key in client.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            return await client.delete(*keys)
        return 0

    # Pub/Sub
    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel."""
        client = await self._get_client()
        return await client.publish(channel, message)

    async def subscribe(self, *channels: str):
        """Subscribe to channels."""
        client = await self._get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub


async def check_dragonfly_health(url: Optional[str] = None) -> DragonflyHealthStatus:
    """Quick health check for DragonflyDB."""
    adapter = DragonflyAdapter(url=url)
    try:
        return await adapter.health_check()
    finally:
        await adapter.close()
