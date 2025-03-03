from typing import Dict, Any
import os

class ScalabilityConfig:
    # Vector Search Configuration
    VECTOR_BATCH_SIZE: int = 100
    VECTOR_CACHE_TTL: int = 3600  # 1 hour
    VECTOR_MAX_CONCURRENT_REQUESTS: int = 50
    VECTOR_RETRY_ATTEMPTS: int = 3
    
    # Memory Management
    MAX_MEMORY_ENTRIES: int = 1000
    MEMORY_CLEANUP_THRESHOLD: float = 0.8  # 80% full
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # per minute
    
    # Connection Pooling
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    
    # Operation Timeouts
    CLASSIFICATION_TIMEOUT: int = 10  # seconds
    CONTEXT_SEARCH_TIMEOUT: int = 10  # seconds
    ANSWER_GENERATION_TIMEOUT: int = 15  # seconds
    VERIFICATION_TIMEOUT: int = 15  # seconds
    
    # Caching Strategy
    CACHE_STRATEGY: Dict[str, Any] = {
        "embeddings": {
            "ttl": 24 * 3600,  # 24 hours
            "max_size": 10000
        },
        "search_results": {
            "ttl": 3600,  # 1 hour
            "max_size": 5000
        },
        "classifications": {
            "ttl": 12 * 3600,  # 12 hours
            "max_size": 1000
        }
    }
    
    # Batch Processing
    MIN_BATCH_SIZE: int = 10
    MAX_BATCH_SIZE: int = 1000
    BATCH_TIMEOUT: float = 0.5  # seconds
    
    @classmethod
    def get_cache_config(cls, cache_type: str) -> Dict[str, Any]:
        return cls.CACHE_STRATEGY.get(cache_type, {
            "ttl": 3600,
            "max_size": 1000
        }) 