"""Caching layer for SQL synthesis agent performance optimization.

This module provides intelligent caching for SQL queries, generation results,
and schema information with TTL, LRU eviction, and cache warming capabilities.
"""

import hashlib
import logging
import time
import threading
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import json
import pickle

from .performance_optimizer import adaptive_cache_strategy, performance_monitor
from .exceptions import CacheError, create_error_context


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                logger.debug("Cache entry expired: %s", key)
                return None
            
            # Update access info and move to end (most recent)
            entry.touch()
            self._cache.move_to_end(key)
            self._hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes,
            )
            
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            
            # Evict if necessary
            self._evict_if_needed()
            
            logger.debug("Cache entry added: %s (size: %d bytes)", key, size_bytes)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("Cache entry deleted: %s", key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.info("Cache cleared")
    
    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if cache is full."""
        while len(self._cache) > self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1
            logger.debug("Cache entry evicted (LRU): %s", oldest_key)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug("Cleaned up %d expired cache entries", len(expired_keys))
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "total_size_bytes": total_size,
                "avg_entry_size": total_size / len(self._cache) if self._cache else 0,
            }


class QueryCache:
    """Specialized cache for SQL query results and metadata."""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self.cache = LRUCache(max_size, default_ttl)
        self.schema_cache = LRUCache(max_size=100, default_ttl=7200)  # 2 hours for schema
        self.generation_cache = LRUCache(max_size=1000, default_ttl=3600)  # 1 hour for generations
        
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        # Create deterministic key from parameters
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]  # Use SHA-256 instead of MD5
        return f"{prefix}:{key_hash}"
    
    def get_query_result(self, sql_query: str, parameters: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        key = self._generate_key("query", sql=sql_query, params=parameters or {})
        return self.cache.get(key)
    
    def put_query_result(self, sql_query: str, result: Dict[str, Any], parameters: Optional[Dict] = None, ttl: Optional[int] = None) -> None:
        """Cache query result."""
        key = self._generate_key("query", sql=sql_query, params=parameters or {})
        self.cache.put(key, result, ttl)
    
    def get_schema_info(self, database_url: str, table_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached schema information."""
        key = self._generate_key("schema", db=database_url, table=table_name or "all")
        return self.schema_cache.get(key)
    
    def put_schema_info(self, database_url: str, schema_info: Dict[str, Any], table_name: Optional[str] = None) -> None:
        """Cache schema information."""
        key = self._generate_key("schema", db=database_url, table=table_name or "all")
        self.schema_cache.put(key, schema_info)
    
    def get_sql_generation(self, natural_language_query: str, database_context: str) -> Optional[Dict[str, Any]]:
        """Get cached SQL generation result."""
        key = self._generate_key("generation", query=natural_language_query, context=database_context)
        return self.generation_cache.get(key)
    
    def put_sql_generation(self, natural_language_query: str, database_context: str, result: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Cache SQL generation result."""
        key = self._generate_key("generation", query=natural_language_query, context=database_context)
        self.generation_cache.put(key, result, ttl)
    
    def invalidate_schema_cache(self, database_url: str) -> None:
        """Invalidate schema cache for a database."""
        # Remove all schema entries for this database
        with self.schema_cache._lock:
            keys_to_remove = []
            for key, entry in self.schema_cache._cache.items():
                if key.startswith("schema:"):
                    try:
                        # Extract database from cached entry
                        if database_url in str(entry.value):
                            keys_to_remove.append(key)
                    except Exception as e:
                        logger.debug("Error checking cache entry %s: %s", key, str(e))
            
            for key in keys_to_remove:
                self.schema_cache.delete(key)
            
            logger.info("Invalidated %d schema cache entries for database", len(keys_to_remove))
    
    def warm_cache(self, common_queries: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Pre-populate cache with common queries."""
        logger.info("Warming cache with %d common queries", len(common_queries))
        
        for sql_query, expected_result in common_queries:
            key = self._generate_key("query", sql=sql_query, params={})
            if not self.cache.get(key):  # Only cache if not already present
                self.cache.put(key, expected_result)
        
        logger.info("Cache warming completed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "query_cache": self.cache.get_stats(),
            "schema_cache": self.schema_cache.get_stats(),
            "generation_cache": self.generation_cache.get_stats(),
        }
    
    def cleanup_all(self) -> Dict[str, int]:
        """Clean up expired entries from all caches."""
        return {
            "query_expired": self.cache.cleanup_expired(),
            "schema_expired": self.schema_cache.cleanup_expired(),
            "generation_expired": self.generation_cache.cleanup_expired(),
        }


class CacheManager:
    """Global cache manager with automatic cleanup."""
    
    def __init__(self):
        self.query_cache = QueryCache()
        self._cleanup_thread = None
        self._cleanup_interval = 300  # 5 minutes
        self._shutdown_event = threading.Event()
        
    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._shutdown_event.clear()
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
            logger.info("Cache cleanup thread started")
    
    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._shutdown_event.set()
            self._cleanup_thread.join(timeout=5)
            logger.info("Cache cleanup thread stopped")
    
    def _cleanup_worker(self) -> None:
        """Background worker for cache cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up expired entries
                cleanup_stats = self.query_cache.cleanup_all()
                total_cleaned = sum(cleanup_stats.values())
                
                if total_cleaned > 0:
                    logger.debug("Cleaned up %d expired cache entries", total_cleaned)
                
                # Wait for next cleanup cycle
                self._shutdown_event.wait(self._cleanup_interval)
                
            except Exception as e:
                logger.error("Error in cache cleanup worker: %s", str(e))
                # Wait before retrying
                self._shutdown_event.wait(60)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.query_cache.get_cache_stats()
        
        # Calculate total statistics
        total_hits = sum(cache_stats["hits"] for cache_stats in stats.values())
        total_misses = sum(cache_stats["misses"] for cache_stats in stats.values())
        total_requests = total_hits + total_misses
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "individual_caches": stats,
            "overall": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "overall_hit_rate": overall_hit_rate,
                "cleanup_thread_active": self._cleanup_thread is not None and self._cleanup_thread.is_alive(),
            }
        }


# Global cache manager instance
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    return cache_manager


# Caching decorators
def cache_query_result(ttl: int = 3600):
    """Decorator to cache query results."""
    def decorator(func):
        def wrapper(self, sql_query: str, parameters: Optional[Dict] = None, **kwargs):
            # Try to get from cache first
            cached_result = cache_manager.query_cache.get_query_result(sql_query, parameters)
            if cached_result is not None:
                logger.debug("Cache hit for query: %s", sql_query[:50])
                return cached_result
            
            # Execute function and cache result
            result = func(self, sql_query, parameters, **kwargs)
            
            # Only cache successful results
            if result.get("success"):
                cache_manager.query_cache.put_query_result(sql_query, result, parameters, ttl)
                logger.debug("Cached query result: %s", sql_query[:50])
            
            return result
        return wrapper
    return decorator


def cache_generation_result(ttl: int = 3600):
    """Decorator to cache SQL generation results."""
    def decorator(func):
        def wrapper(self, natural_language_query: str, **kwargs):
            # Create context key from database info
            database_context = str(getattr(self, 'database_manager', {}).get('db_type', 'unknown'))
            
            # Try to get from cache first
            cached_result = cache_manager.query_cache.get_sql_generation(natural_language_query, database_context)
            if cached_result is not None:
                logger.debug("Cache hit for generation: %s", natural_language_query[:50])
                return cached_result
            
            # Execute function and cache result
            result = func(self, natural_language_query, **kwargs)
            
            # Only cache successful results
            if result.get("success"):
                cache_manager.query_cache.put_sql_generation(natural_language_query, database_context, result, ttl)
                logger.debug("Cached generation result: %s", natural_language_query[:50])
            
            return result
        return wrapper
    return decorator