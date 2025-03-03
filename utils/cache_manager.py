from typing import Any, Dict, Optional
import time
from collections import OrderedDict
from threading import Lock
from config import ScalabilityConfig

class CacheEntry:
    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.expiry = time.time() + ttl
    
    def is_expired(self) -> bool:
        return time.time() > self.expiry

class LRUCache:
    def __init__(self, max_size: int, ttl: int):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return entry.value
    
    def set(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = CacheEntry(value, self.ttl)
    
    def clear_expired(self) -> None:
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, v in self.cache.items()
                if current_time > v.expiry
            ]
            for k in expired_keys:
                del self.cache[k]

class CacheManager:
    _instance = None
    _caches: Dict[str, LRUCache] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialize_caches()
        return cls._instance
    
    def _initialize_caches(self) -> None:
        """Initialize different cache types with their configurations."""
        for cache_type, config in ScalabilityConfig.CACHE_STRATEGY.items():
            self._caches[cache_type] = LRUCache(
                max_size=config["max_size"],
                ttl=config["ttl"]
            )
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from specified cache type."""
        if cache_type not in self._caches:
            return None
        return self._caches[cache_type].get(key)
    
    def set(self, cache_type: str, key: str, value: Any) -> None:
        """Set value in specified cache type."""
        if cache_type in self._caches:
            self._caches[cache_type].set(key, value)
    
    def clear_expired(self, cache_type: Optional[str] = None) -> None:
        """Clear expired entries from specified or all caches."""
        if cache_type:
            if cache_type in self._caches:
                self._caches[cache_type].clear_expired()
        else:
            for cache in self._caches.values():
                cache.clear_expired()
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about cache usage."""
        return {
            cache_type: {
                "size": len(cache.cache),
                "max_size": cache.max_size
            }
            for cache_type, cache in self._caches.items()
        } 