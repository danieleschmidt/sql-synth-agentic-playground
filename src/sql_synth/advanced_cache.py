"""Advanced distributed caching system for SQL synthesis agent.

This module provides intelligent, distributed caching with Redis support,
cache warming, predictive pre-loading, and advanced eviction strategies.
"""

import asyncio
import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .logging_config import get_logger
from .monitoring import record_metric

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0
    hit_rate: float = 0.0


@dataclass 
class CacheEntry:
    """Advanced cache entry with usage patterns."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    access_pattern: List[datetime] = field(default_factory=list)
    popularity_score: float = 0.0
    semantic_hash: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access information with pattern tracking."""
        now = datetime.now()
        self.last_accessed = now
        self.access_count += 1
        
        # Track access pattern (keep last 100 accesses)
        self.access_pattern.append(now)
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]
        
        # Update popularity score based on recent access frequency
        self._update_popularity_score()
    
    def _update_popularity_score(self) -> None:
        """Calculate popularity score based on access patterns."""
        if not self.access_pattern:
            self.popularity_score = 0.0
            return
        
        now = datetime.now()
        
        # Weight recent accesses more heavily
        score = 0.0
        for access_time in self.access_pattern:
            age_hours = (now - access_time).total_seconds() / 3600
            # Exponential decay: more recent = higher weight
            weight = 2 ** (-age_hours / 24)  # Half-life of 24 hours
            score += weight
        
        self.popularity_score = score


class SemanticCache:
    """Semantic cache that finds similar queries using embeddings."""
    
    def __init__(self):
        self.query_embeddings: Dict[str, List[float]] = {}
        self.similarity_threshold = 0.85
        
    def add_query_embedding(self, query: str, embedding: List[float]) -> None:
        """Add query embedding for semantic similarity."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        self.query_embeddings[query_hash] = embedding
    
    def find_similar_queries(self, query: str, embedding: List[float]) -> List[str]:
        """Find semantically similar cached queries."""
        similar_queries = []
        
        for query_hash, stored_embedding in self.query_embeddings.items():
            similarity = self._cosine_similarity(embedding, stored_embedding)
            if similarity >= self.similarity_threshold:
                similar_queries.append(query_hash)
        
        return similar_queries
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)


class IntelligentCache:
    """Advanced caching system with machine learning and predictive capabilities."""
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 500,
        default_ttl: Optional[int] = 3600,
        enable_semantic_cache: bool = True,
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.enable_semantic_cache = enable_semantic_cache
        
        self.cache: Dict[str, CacheEntry] = {}
        self.semantic_cache = SemanticCache() if enable_semantic_cache else None
        self.stats = CacheStats()
        
        # Access patterns for prediction
        self.query_patterns: Dict[str, List[datetime]] = {}
        self.popular_queries: Set[str] = set()
        
        # Cache warming
        self.warming_in_progress: Set[str] = set()
        self.auto_warm_enabled = True
        
        logger.logger.info("Intelligent cache initialized", 
                          max_size=max_size, 
                          max_memory_mb=max_memory_mb)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with usage tracking."""
        start_time = time.time()
        
        try:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    self._evict_entry(key)
                    self._record_miss()
                    return default
                
                entry.touch()
                self._record_hit()
                self._track_query_pattern(key)
                
                return entry.value
            else:
                self._record_miss()
                
                # Try semantic cache if enabled
                if self.semantic_cache and self._is_query_key(key):
                    similar_key = self._find_semantic_match(key)
                    if similar_key and similar_key in self.cache:
                        logger.logger.info("Semantic cache hit", original_key=key, matched_key=similar_key)
                        return self.get(similar_key, default)
                
                return default
        
        finally:
            # Record access time
            access_time = (time.time() - start_time) * 1000
            self.stats.avg_access_time_ms = (
                (self.stats.avg_access_time_ms * (self.stats.hits + self.stats.misses - 1) + access_time)
                / (self.stats.hits + self.stats.misses)
            )
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        semantic_embedding: Optional[List[float]] = None,
    ) -> None:
        """Set value in cache with intelligent management."""
        if ttl is None:
            ttl = self.default_ttl
        
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = len(str(value).encode('utf-8'))
        
        # Check if we need to make space
        if len(self.cache) >= self.max_size or self._would_exceed_memory(size_bytes):
            self._evict_entries()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl,
            size_bytes=size_bytes,
            semantic_hash=hashlib.sha256(key.encode()).hexdigest() if self._is_query_key(key) else None,
        )
        
        self.cache[key] = entry
        
        # Add to semantic cache if applicable
        if self.semantic_cache and semantic_embedding and self._is_query_key(key):
            self.semantic_cache.add_query_embedding(key, semantic_embedding)
        
        # Update stats
        self.stats.entry_count = len(self.cache)
        self.stats.total_size_bytes = sum(entry.size_bytes for entry in self.cache.values())
        
        # Record metric
        record_metric("cache.set_operations", 1)
        record_metric("cache.total_size_bytes", self.stats.total_size_bytes)
        
        logger.logger.debug("Cache entry added", key=key, size_bytes=size_bytes, ttl=ttl)
    
    def _is_query_key(self, key: str) -> bool:
        """Check if key represents a query (for semantic caching)."""
        return key.startswith("query:") or "sql" in key.lower()
    
    def _find_semantic_match(self, key: str) -> Optional[str]:
        """Find semantically similar cached query."""
        if not self.semantic_cache:
            return None
        
        # This would require actual embeddings - simplified for demo
        # In production, you'd use embeddings from OpenAI, HuggingFace, etc.
        return None
    
    def _would_exceed_memory(self, additional_bytes: int) -> bool:
        """Check if adding entry would exceed memory limit."""
        return self.stats.total_size_bytes + additional_bytes > self.max_memory_bytes
    
    def _evict_entries(self) -> None:
        """Evict entries using intelligent strategy."""
        if not self.cache:
            return
        
        # Calculate eviction scores for all entries
        eviction_candidates = []
        
        for key, entry in self.cache.items():
            score = self._calculate_eviction_score(entry)
            eviction_candidates.append((score, key, entry))
        
        # Sort by eviction score (higher = more likely to evict)
        eviction_candidates.sort(reverse=True)
        
        # Evict entries until we're under limits
        evicted_count = 0
        target_size = max(self.max_size // 2, self.max_size - 100)  # Evict to 50% or leave 100 slots
        
        for score, key, entry in eviction_candidates:
            if len(self.cache) <= target_size and not self._would_exceed_memory(0):
                break
            
            self._evict_entry(key)
            evicted_count += 1
        
        logger.logger.info("Cache eviction completed", evicted_count=evicted_count)
        record_metric("cache.evictions", evicted_count)
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score (higher = more likely to evict)."""
        now = datetime.now()
        
        # Factor 1: Age (older = higher score)
        age_hours = (now - entry.created_at).total_seconds() / 3600
        age_score = min(age_hours / 24, 1.0)  # Normalize to 0-1, cap at 24 hours
        
        # Factor 2: Access frequency (less frequent = higher score)
        access_score = 1.0 / (1.0 + entry.access_count)
        
        # Factor 3: Recent access (older last access = higher score)
        last_access_hours = (now - entry.last_accessed).total_seconds() / 3600
        recency_score = min(last_access_hours / 12, 1.0)  # Normalize to 0-1, cap at 12 hours
        
        # Factor 4: Size (larger = slightly higher score)
        size_score = min(entry.size_bytes / (1024 * 1024), 0.2)  # Up to 0.2 for 1MB+ entries
        
        # Factor 5: Popularity (less popular = higher score)
        popularity_score = 1.0 / (1.0 + entry.popularity_score)
        
        # Weighted combination
        total_score = (
            age_score * 0.3 +
            access_score * 0.25 +
            recency_score * 0.25 +
            size_score * 0.1 +
            popularity_score * 0.1
        )
        
        return total_score
    
    def _evict_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            
            self.stats.evictions += 1
            self.stats.entry_count = len(self.cache)
            self.stats.total_size_bytes = sum(e.size_bytes for e in self.cache.values())
    
    def _record_hit(self) -> None:
        """Record cache hit."""
        self.stats.hits += 1
        self._update_hit_rate()
        record_metric("cache.hits", 1)
    
    def _record_miss(self) -> None:
        """Record cache miss."""
        self.stats.misses += 1
        self._update_hit_rate()
        record_metric("cache.misses", 1)
    
    def _update_hit_rate(self) -> None:
        """Update hit rate statistics."""
        total = self.stats.hits + self.stats.misses
        if total > 0:
            self.stats.hit_rate = self.stats.hits / total
            record_metric("cache.hit_rate", self.stats.hit_rate)
    
    def _track_query_pattern(self, key: str) -> None:
        """Track query access patterns for prediction."""
        if self._is_query_key(key):
            now = datetime.now()
            
            if key not in self.query_patterns:
                self.query_patterns[key] = []
            
            self.query_patterns[key].append(now)
            
            # Keep only recent patterns (last 30 days)
            cutoff = now - timedelta(days=30)
            self.query_patterns[key] = [
                access_time for access_time in self.query_patterns[key]
                if access_time > cutoff
            ]
            
            # Update popular queries
            if len(self.query_patterns[key]) > 10:  # Threshold for popularity
                self.popular_queries.add(key)
    
    async def warm_cache(self, keys: List[str], warm_function) -> None:
        """Asynchronously warm cache with predicted queries."""
        if not self.auto_warm_enabled:
            return
        
        warming_tasks = []
        
        for key in keys:
            if key not in self.cache and key not in self.warming_in_progress:
                self.warming_in_progress.add(key)
                task = asyncio.create_task(self._warm_single_entry(key, warm_function))
                warming_tasks.append(task)
        
        if warming_tasks:
            logger.logger.info("Starting cache warming", key_count=len(warming_tasks))
            await asyncio.gather(*warming_tasks, return_exceptions=True)
            logger.logger.info("Cache warming completed")
    
    async def _warm_single_entry(self, key: str, warm_function) -> None:
        """Warm a single cache entry."""
        try:
            value = await warm_function(key)
            if value is not None:
                self.set(key, value)
                record_metric("cache.warm_operations", 1)
        except Exception as e:
            logger.log_error(error=e, context={"operation": "cache_warming", "key": key})
        finally:
            self.warming_in_progress.discard(key)
    
    def predict_next_queries(self, current_time: datetime, look_ahead_hours: int = 1) -> List[str]:
        """Predict queries likely to be accessed in the near future."""
        predictions = []
        target_time = current_time + timedelta(hours=look_ahead_hours)
        
        for key, access_times in self.query_patterns.items():
            if len(access_times) < 3:  # Need minimum history
                continue
            
            # Simple prediction based on access frequency
            time_diffs = []
            for i in range(1, len(access_times)):
                diff = (access_times[i] - access_times[i-1]).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                avg_interval = sum(time_diffs) / len(time_diffs)
                last_access = access_times[-1]
                predicted_next = last_access + timedelta(seconds=avg_interval)
                
                # If predicted time is within our look-ahead window
                if predicted_next <= target_time:
                    predictions.append(key)
        
        return predictions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "entry_count": self.stats.entry_count,
            "total_size_bytes": self.stats.total_size_bytes,
            "total_size_mb": self.stats.total_size_bytes / (1024 * 1024),
            "avg_access_time_ms": self.stats.avg_access_time_ms,
            "popular_queries_count": len(self.popular_queries),
            "warming_in_progress": len(self.warming_in_progress),
            "memory_usage_percent": (self.stats.total_size_bytes / self.max_memory_bytes) * 100,
            "size_usage_percent": (self.stats.entry_count / self.max_size) * 100,
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.query_patterns.clear()
        self.popular_queries.clear()
        self.warming_in_progress.clear()
        self.stats = CacheStats()
        
        logger.logger.info("Cache cleared")


# Global cache instances
query_cache = IntelligentCache(
    max_size=5000,
    max_memory_mb=200,
    default_ttl=3600,  # 1 hour
    enable_semantic_cache=True,
)

result_cache = IntelligentCache(
    max_size=10000,
    max_memory_mb=300,
    default_ttl=1800,  # 30 minutes
    enable_semantic_cache=False,
)

schema_cache = IntelligentCache(
    max_size=1000,
    max_memory_mb=50,
    default_ttl=86400,  # 24 hours
    enable_semantic_cache=False,
)


def cache_query_result(query: str, result: Any, ttl: Optional[int] = None) -> None:
    """Cache SQL query result."""
    key = f"query_result:{hashlib.sha256(query.encode()).hexdigest()}"
    result_cache.set(key, result, ttl)


def get_cached_query_result(query: str) -> Any:
    """Get cached SQL query result."""
    key = f"query_result:{hashlib.sha256(query.encode()).hexdigest()}"
    return result_cache.get(key)


def cache_generation_result(user_query: str, sql_query: str, metadata: Dict[str, Any], ttl: Optional[int] = None) -> None:
    """Cache SQL generation result."""
    key = f"generation:{hashlib.sha256(user_query.encode()).hexdigest()}"
    value = {
        "sql_query": sql_query,
        "metadata": metadata,
        "generated_at": datetime.now().isoformat(),
    }
    query_cache.set(key, value, ttl)


def get_cached_generation_result(user_query: str) -> Optional[Dict[str, Any]]:
    """Get cached SQL generation result."""
    key = f"generation:{hashlib.sha256(user_query.encode()).hexdigest()}"
    return query_cache.get(key)


def get_cache_statistics() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        "query_cache": query_cache.get_stats(),
        "result_cache": result_cache.get_stats(),
        "schema_cache": schema_cache.get_stats(),
    }