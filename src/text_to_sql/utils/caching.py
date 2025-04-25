"""
Caching Module

This module provides caching utilities for database schemas, query results,
and LLM responses to improve performance.
"""

import functools
import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from text_to_sql.utils.config_types import DatabaseConfig

logger = logging.getLogger(__name__)

class Cache:
    """Base cache class that provides common caching functionality."""
    
    def __init__(
        self,
        name: str,
        max_size: int = 100,
        ttl: int = 3600,  # Time-to-live in seconds (1 hour default)
        persist: bool = False,
        cache_dir: str = ".cache"
    ):
        """
        Initialize the cache.
        
        Args:
            name: Cache name
            max_size: Maximum number of items in the cache
            ttl: Time-to-live in seconds (0 means no expiration)
            persist: Whether to persist the cache to disk
            cache_dir: Directory for persistent cache files
        """
        self.name = name
        self.max_size = max_size
        self.ttl = ttl
        self.persist = persist
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f"{name}.cache"
        
        # Create cache directory if needed
        if self.persist:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = {}
        self.access_times = {}
        
        # Load cache from disk if persisting
        if self.persist and self.cache_file.exists():
            self.load_cache()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            Cached value or default
        """
        key_hash = self._hash_key(key)
        
        # Check if key exists
        if key_hash not in self.cache:
            return default
        
        # Check if key has expired
        if self.ttl > 0:
            last_access = self.access_times.get(key_hash, 0)
            if time.time() - last_access > self.ttl:
                # Remove expired item
                self._remove(key_hash)
                return default
        
        # Update access time
        self.access_times[key_hash] = time.time()
        
        return self.cache[key_hash]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        key_hash = self._hash_key(key)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size and key_hash not in self.cache:
            # Remove least recently used item
            self._remove_lru()
        
        # Set value and access time
        self.cache[key_hash] = value
        self.access_times[key_hash] = time.time()
        
        # Save cache if persisting
        if self.persist:
            self.save_cache()
    
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        key_hash = self._hash_key(key)
        self._remove(key_hash)
        
        # Save cache if persisting
        if self.persist:
            self.save_cache()
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.access_times = {}
        
        # Remove cache file if persisting
        if self.persist and self.cache_file.exists():
            self.cache_file.unlink()
    
    def save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump((self.cache, self.access_times), f)
            
            logger.debug(f"Cache '{self.name}' saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache '{self.name}': {e}")
    
    def load_cache(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_file, "rb") as f:
                self.cache, self.access_times = pickle.load(f)
            
            # Remove expired items
            if self.ttl > 0:
                now = time.time()
                expired_keys = [
                    key for key, access_time in self.access_times.items()
                    if now - access_time > self.ttl
                ]
                
                for key in expired_keys:
                    self._remove(key)
            
            logger.debug(f"Cache '{self.name}' loaded from {self.cache_file}")
        except Exception as e:
            logger.error(f"Error loading cache '{self.name}': {e}")
            self.cache = {}
            self.access_times = {}
    
    def _hash_key(self, key: str) -> str:
        """
        Hash a key for consistent storage.
        
        Args:
            key: Cache key
            
        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    
    def _remove(self, key_hash: str) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key_hash: Hashed key
        """
        if key_hash in self.cache:
            del self.cache[key_hash]
        
        if key_hash in self.access_times:
            del self.access_times[key_hash]
    
    def _remove_lru(self) -> None:
        """Remove the least recently used item from the cache."""
        if not self.access_times:
            return
        
        # Find the key with the oldest access time
        oldest_key = min(self.access_times, key=self.access_times.get)
        self._remove(oldest_key)


class SchemaCache(Cache):
    """Cache for database schemas."""
    
    def __init__(
        self,
        max_size: int = 10,
        ttl: int = 3600 * 24,  # 24 hours
        persist: bool = True,
        cache_dir: str = ".cache"
    ):
        """
        Initialize the schema cache.
        
        Args:
            max_size: Maximum number of schemas to cache
            ttl: Time-to-live in seconds
            persist: Whether to persist the cache to disk
            cache_dir: Directory for persistent cache files
        """
        super().__init__("schema", max_size, ttl, persist, cache_dir)
    
    def get_schema(
        self,
        db_type: str,
        connection_params: Union[Dict[str, Any], DatabaseConfig]
    ) -> Optional[Dict[str, Any]]:
        """
        Get a schema from the cache.
        
        Args:
            db_type: Database type
            connection_params: Connection parameters or DatabaseConfig
            
        Returns:
            Cached schema or None if not found
        """
        # Convert DatabaseConfig to dict if needed
        if isinstance(connection_params, DatabaseConfig):
            connection_params = connection_params.get_connection_params()
        
        # Create a key from db_type and connection_params
        key = f"{db_type}:{json.dumps(connection_params, sort_keys=True)}"
        return self.get(key)
    
    def set_schema(
        self,
        db_type: str,
        connection_params: Union[Dict[str, Any], DatabaseConfig],
        schema: Dict[str, Any]
    ) -> None:
        """
        Set a schema in the cache.
        
        Args:
            db_type: Database type
            connection_params: Connection parameters or DatabaseConfig
            schema: Schema to cache
        """
        # Convert DatabaseConfig to dict if needed
        if isinstance(connection_params, DatabaseConfig):
            connection_params = connection_params.get_connection_params()
        
        # Create a key from db_type and connection_params
        key = f"{db_type}:{json.dumps(connection_params, sort_keys=True)}"
        self.set(key, schema)


class QueryCache(Cache):
    """Cache for query results."""
    
    def __init__(
        self,
        max_size: int = 100,
        ttl: int = 3600,  # 1 hour
        persist: bool = True,
        cache_dir: str = ".cache"
    ):
        """
        Initialize the query cache.
        
        Args:
            max_size: Maximum number of query results to cache
            ttl: Time-to-live in seconds
            persist: Whether to persist the cache to disk
            cache_dir: Directory for persistent cache files
        """
        super().__init__("query", max_size, ttl, persist, cache_dir)
    
    def get_results(
        self,
        query: str,
        db_type: str,
        connection_params: Union[Dict[str, Any], DatabaseConfig]
    ) -> Optional[Tuple[List[Dict[str, Any]], Optional[str]]]:
        """
        Get query results from the cache.
        
        Args:
            query: SQL query
            db_type: Database type
            connection_params: Connection parameters or DatabaseConfig
            
        Returns:
            Cached query results or None if not found
        """
        # Convert DatabaseConfig to dict if needed
        if isinstance(connection_params, DatabaseConfig):
            connection_params = connection_params.get_connection_params()
        
        # Create a key from query, db_type, and connection_params
        key = f"{query}:{db_type}:{json.dumps(connection_params, sort_keys=True)}"
        return self.get(key)
    
    def set_results(
        self,
        query: str,
        db_type: str,
        connection_params: Union[Dict[str, Any], DatabaseConfig],
        results: Tuple[List[Dict[str, Any]], Optional[str]]
    ) -> None:
        """
        Set query results in the cache.
        
        Args:
            query: SQL query
            db_type: Database type
            connection_params: Connection parameters or DatabaseConfig
            results: Query results to cache
        """
        # Convert DatabaseConfig to dict if needed
        if isinstance(connection_params, DatabaseConfig):
            connection_params = connection_params.get_connection_params()
        
        # Create a key from query, db_type, and connection_params
        key = f"{query}:{db_type}:{json.dumps(connection_params, sort_keys=True)}"
        self.set(key, results)


class LLMCache(Cache):
    """Cache for LLM responses."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600 * 24,  # 24 hours
        persist: bool = True,
        cache_dir: str = ".cache"
    ):
        """
        Initialize the LLM cache.
        
        Args:
            max_size: Maximum number of LLM responses to cache
            ttl: Time-to-live in seconds
            persist: Whether to persist the cache to disk
            cache_dir: Directory for persistent cache files
        """
        super().__init__("llm", max_size, ttl, persist, cache_dir)
    
    def get_response(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0
    ) -> Optional[str]:
        """
        Get an LLM response from the cache.
        
        Args:
            model: LLM model
            prompt: Prompt sent to the LLM
            temperature: Temperature parameter
            
        Returns:
            Cached LLM response or None if not found
        """
        # Create a key from model, prompt, and temperature
        key = f"{model}:{prompt}:{temperature}"
        return self.get(key)
    
    def set_response(
        self,
        model: str,
        prompt: str,
        temperature: float,
        response: str
    ) -> None:
        """
        Set an LLM response in the cache.
        
        Args:
            model: LLM model
            prompt: Prompt sent to the LLM
            temperature: Temperature parameter
            response: LLM response to cache
        """
        # Create a key from model, prompt, and temperature
        key = f"{model}:{prompt}:{temperature}"
        self.set(key, response)


def cached(
    cache: Cache,
    key_fn: Callable = None,
    ttl: Optional[int] = None
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache instance
        key_fn: Function to generate cache key from function arguments
        ttl: Optional TTL override for this function
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default key is function name and arguments
                key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            if ttl is not None:
                # Save current TTL
                old_ttl = cache.ttl
                # Set new TTL
                cache.ttl = ttl
                # Cache result
                cache.set(key, result)
                # Restore old TTL
                cache.ttl = old_ttl
            else:
                cache.set(key, result)
            
            return result
        return wrapper
    return decorator


# Create cache instances
schema_cache = SchemaCache()
query_cache = QueryCache()
llm_cache = LLMCache()


# Decorator for caching database schemas
def cached_schema(func):
    """Decorator for caching database schemas."""
    @functools.wraps(func)
    def wrapper(self, refresh=False, *args, **kwargs):
        # Skip cache if refresh is True
        if refresh:
            return func(self, refresh, *args, **kwargs)
        
        # Get cached schema
        db_type = self.get_database_type()
        cached_schema = schema_cache.get_schema(db_type, self.connection_params)
        
        if cached_schema is not None:
            logger.debug(f"Using cached schema for {db_type}")
            return cached_schema
        
        # Get schema
        schema = func(self, refresh, *args, **kwargs)
        
        # Cache schema
        schema_cache.set_schema(db_type, self.connection_params, schema)
        
        return schema
    return wrapper


# Decorator for caching query results
def cached_query(func):
    """Decorator for caching query results."""
    @functools.wraps(func)
    def wrapper(self, query, params=None, *args, **kwargs):
        # Skip cache for non-SELECT queries
        if not query.strip().upper().startswith("SELECT"):
            return func(self, query, params, *args, **kwargs)
        
        # Get cached results
        db_type = self.get_database_type()
        cached_results = query_cache.get_results(query, db_type, self.connection_params)
        
        if cached_results is not None:
            logger.debug(f"Using cached results for query: {query}")
            return cached_results
        
        # Execute query
        results = func(self, query, params, *args, **kwargs)
        
        # Only cache successful queries without errors
        if results[1] is None:
            # Cache results
            query_cache.set_results(query, db_type, self.connection_params, results)
        
        return results
    return wrapper


# Decorator for caching LLM responses
def cached_llm(func):
    """Decorator for caching LLM responses."""
    @functools.wraps(func)
    def wrapper(self, prompt, *args, **kwargs):
        # Get cached response
        model = self.model
        temperature = getattr(self, "temperature", 0.0)
        cached_response = llm_cache.get_response(model, prompt, temperature)
        
        if cached_response is not None:
            logger.debug(f"Using cached LLM response for model: {model}")
            return cached_response
        
        # Get response from LLM
        response = func(self, prompt, *args, **kwargs)
        
        # Cache response
        llm_cache.set_response(model, prompt, temperature, response)
        
        return response
    return wrapper