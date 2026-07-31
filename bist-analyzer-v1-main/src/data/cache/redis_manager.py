# src/data/cache/redis_manager.py
import json
import logging
from typing import Optional, Any, Dict
import time

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis paketi yüklü değil. In-memory cache kullanılacak.")


class RedisManager:
    """
    Önbellek yöneticisi.
    Redis varsa kullanır, yoksa otomatik in-memory fallback.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 300
    ):
        self._redis = None
        self._memory_cache: Dict[str, Any] = {}
        self._memory_expiry: Dict[str, float] = {}
        self._default_ttl = default_ttl
        self._redis_config = {
            "host": host, "port": port, "db": db, "password": password
        }
        
        if REDIS_AVAILABLE:
            try:
                self._redis = aioredis.Redis(
                    host=host, port=port, db=db, password=password,
                    decode_responses=True, socket_connect_timeout=3
                )
                logger.info(f"Redis bağlantısı kuruldu: {host}")
            except Exception as e:
                logger.warning(f"Redis bağlantısı başarısız: {e}. Memory fallback aktif.")
                self._redis = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Önbellekten veri oku"""
        # Redis dene
        if self._redis:
            try:
                value = await self._redis.get(key)
                if value:
                    return json.loads(value)
            except Exception:
                pass
        
        # Memory fallback
        if key in self._memory_cache:
            expiry = self._memory_expiry.get(key, 0)
            if expiry > time.time():
                return self._memory_cache[key]
            else:
                del self._memory_cache[key]
                del self._memory_expiry[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Önbelleğe veri yaz"""
        if ttl is None:
            ttl = self._default_ttl
        
        # Redis dene
        if self._redis:
            try:
                serialized = json.dumps(value, default=str)
                await self._redis.setex(key, ttl, serialized)
                return True
            except Exception:
                pass
        
        # Memory fallback
        self._memory_cache[key] = value
        self._memory_expiry[key] = time.time() + ttl
        return True
    
    async def delete(self, key: str) -> bool:
        """Önbellekten sil"""
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception:
                pass
        
        if key in self._memory_cache:
            del self._memory_cache[key]
            if key in self._memory_expiry:
                del self._memory_expiry[key]
        return True
    
    async def clear_all(self) -> bool:
        """Tüm önbelleği temizle"""
        if self._redis:
            try:
                await self._redis.flushdb()
            except Exception:
                pass
        
        self._memory_cache.clear()
        self._memory_expiry.clear()
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Önbellek istatistikleri"""
        stats = {
            "redis_available": self._redis is not None,
            "memory_cache_size": len(self._memory_cache),
        }
        
        if self._redis:
            try:
                info = await self._redis.info("stats")
                stats["redis_keys"] = info.get("keyspace_hits", 0)
            except Exception:
                pass
        
        return stats