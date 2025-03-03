from typing import Any, Dict, List
import time
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds

class VectorCache:
    def __init__(self, ttl: int = 3600):  # Default TTL: 1 hour
        self._cache: Dict[str, CacheEntry] = {}
        self.default_ttl = ttl
    
    def get(self, key: str) -> Any:
        """Get item from cache if it exists and hasn't expired"""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                return entry.data
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set item in cache with optional TTL"""
        self._cache[key] = CacheEntry(
            data=value,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl
        )
    
    def clear_expired(self) -> None:
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp >= entry.ttl
        ]
        for key in expired_keys:
            del self._cache[key]

@lru_cache(maxsize=1000)
def cache_embedding(text: str) -> List[float]:
    """Cache embeddings for frequently used text"""
    return text  # This will be replaced by actual embedding in the agent

class BatchProcessor:
    def __init__(self, batch_size: int = 10, max_wait: float = 0.1):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.batch: List[Any] = []
        self.last_process_time = time.time()
    
    def add(self, item: Any) -> None:
        """Add item to batch"""
        self.batch.append(item)
    
    def should_process(self) -> bool:
        """Check if batch should be processed"""
        if len(self.batch) >= self.batch_size:
            return True
        if time.time() - self.last_process_time >= self.max_wait:
            return True
        return False
    
    def get_batch(self) -> List[Any]:
        """Get current batch and reset"""
        current_batch = self.batch
        self.batch = []
        self.last_process_time = time.time()
        return current_batch 