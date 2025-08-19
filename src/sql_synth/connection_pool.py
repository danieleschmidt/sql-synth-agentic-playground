"""Advanced connection pooling and resource management for SQL synthesis agent.

This module provides intelligent connection pooling, load balancing,
and resource management for optimal database performance and scalability.
"""

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from typing import Any, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import QueuePool

from .logging_config import get_logger
from .monitoring import record_metric

logger = get_logger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    IDLE = "idle"
    ACTIVE = "active"
    BROKEN = "broken"
    TESTING = "testing"


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""
    created_at: datetime
    last_used: datetime
    total_queries: int = 0
    total_time_seconds: float = 0.0
    error_count: int = 0
    state: ConnectionState = ConnectionState.IDLE


class IntelligentConnectionPool:
    """Advanced connection pool with health monitoring and auto-scaling."""

    def __init__(
        self,
        database_url: str,
        min_connections: int = 2,
        max_connections: int = 20,
        max_idle_time: int = 300,  # 5 minutes
        health_check_interval: int = 60,  # 1 minute
        query_timeout: int = 30,
    ):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.query_timeout = query_timeout

        # Connection management
        self.connections: dict[str, Any] = {}
        self.metrics: dict[str, ConnectionMetrics] = {}
        self.idle_connections: Queue = Queue()
        self.active_connections: dict[str, Any] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Health monitoring
        self.last_health_check = time.time()
        self.health_check_running = False

        # Performance tracking
        self.total_connections_created = 0
        self.total_queries_executed = 0
        self.total_execution_time = 0.0

        # Engine configuration for optimal performance
        self.engine = self._create_optimized_engine()

        # Initialize minimum connections
        self._initialize_pool()

        logger.logger.info(
            "Connection pool initialized",
            min_connections=min_connections,
            max_connections=max_connections,
        )

    def _create_optimized_engine(self) -> Engine:
        """Create SQLAlchemy engine with optimized settings."""
        engine_config = {
            # Connection pool settings
            "poolclass": QueuePool,
            "pool_size": self.max_connections,
            "max_overflow": 10,
            "pool_pre_ping": True,  # Validate connections before use
            "pool_recycle": 3600,   # Recycle connections every hour

            # Connection settings
            "connect_args": {
                "connect_timeout": 30,
                "application_name": "sql_synth_agent",
            },

            # Performance optimizations
            "echo": False,
            "echo_pool": False,
            "future": True,
        }

        # Database-specific optimizations
        if "postgresql" in self.database_url:
            engine_config["connect_args"].update({
                "server_side_cursors": True,
                "prepared_statement_cache_size": 100,
            })
        elif "mysql" in self.database_url:
            engine_config["connect_args"].update({
                "charset": "utf8mb4",
                "autocommit": True,
            })

        return create_engine(self.database_url, **engine_config)

    def _initialize_pool(self) -> None:
        """Initialize the connection pool with minimum connections."""
        for i in range(self.min_connections):
            try:
                conn = self._create_connection()
                self._add_to_idle_pool(conn)
            except Exception as e:
                logger.log_error(
                    error=e,
                    context={"operation": "pool_initialization", "connection_index": i},
                )

    def _create_connection(self) -> Any:
        """Create a new database connection."""
        try:
            conn = self.engine.connect()
            conn_id = f"conn_{self.total_connections_created}_{int(time.time())}"

            with self.lock:
                self.connections[conn_id] = conn
                self.metrics[conn_id] = ConnectionMetrics(
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                )
                self.total_connections_created += 1

            # Test connection
            self._test_connection(conn_id, conn)

            record_metric("connection_pool.connections_created", 1)
            logger.logger.debug("New connection created", connection_id=conn_id)

            return conn_id

        except Exception as e:
            logger.log_error(error=e, context={"operation": "create_connection"})
            raise

    def _test_connection(self, conn_id: str, conn: Any) -> bool:
        """Test if connection is healthy."""
        try:
            with self.lock:
                if conn_id in self.metrics:
                    self.metrics[conn_id].state = ConnectionState.TESTING

            # Simple health check query
            result = conn.execute("SELECT 1").fetchone()

            with self.lock:
                if conn_id in self.metrics:
                    self.metrics[conn_id].state = ConnectionState.IDLE

            return result is not None

        except Exception as e:
            logger.log_error(
                error=e,
                context={"operation": "test_connection", "connection_id": conn_id},
            )

            with self.lock:
                if conn_id in self.metrics:
                    self.metrics[conn_id].state = ConnectionState.BROKEN
                    self.metrics[conn_id].error_count += 1

            return False

    def _add_to_idle_pool(self, conn_id: str) -> None:
        """Add connection to idle pool."""
        with self.lock:
            if conn_id in self.metrics:
                self.metrics[conn_id].state = ConnectionState.IDLE
                self.metrics[conn_id].last_used = datetime.now()

        self.idle_connections.put(conn_id)

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Get a connection from the pool with automatic return."""
        conn_id = None
        conn = None
        start_time = time.time()

        try:
            conn_id = self._acquire_connection()
            conn = self.connections[conn_id]

            with self.lock:
                if conn_id in self.metrics:
                    self.metrics[conn_id].state = ConnectionState.ACTIVE
                    self.metrics[conn_id].last_used = datetime.now()

                self.active_connections[conn_id] = conn

            yield conn

        except Exception as e:
            if conn_id and conn_id in self.metrics:
                with self.lock:
                    self.metrics[conn_id].error_count += 1

            logger.log_error(
                error=e,
                context={"operation": "connection_usage", "connection_id": conn_id},
            )
            raise

        finally:
            if conn_id:
                self._release_connection(conn_id, time.time() - start_time)

    def _acquire_connection(self) -> str:
        """Acquire a connection from the pool."""
        # Try to get idle connection first
        try:
            conn_id = self.idle_connections.get_nowait()

            # Verify connection is still valid
            if conn_id in self.connections:
                conn = self.connections[conn_id]
                if self._test_connection(conn_id, conn):
                    return conn_id
                # Connection is broken, remove it
                self._remove_connection(conn_id)
        except Empty:
            pass

        # No idle connections available, create new one if under limit
        with self.lock:
            if len(self.connections) < self.max_connections:
                return self._create_connection()

        # Wait for a connection to become available
        timeout = 30  # 30 seconds timeout
        start_wait = time.time()

        while time.time() - start_wait < timeout:
            try:
                conn_id = self.idle_connections.get(timeout=1)
                if conn_id in self.connections:
                    return conn_id
            except Empty:
                continue

        msg = "Connection pool exhausted - no connections available"
        raise RuntimeError(msg)

    def _release_connection(self, conn_id: str, execution_time: float) -> None:
        """Release connection back to the pool."""
        with self.lock:
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]

            if conn_id in self.metrics:
                metrics = self.metrics[conn_id]
                metrics.total_queries += 1
                metrics.total_time_seconds += execution_time

                # Check if connection should be retired
                if (metrics.error_count > 5 or
                    metrics.total_queries > 1000 or
                    (datetime.now() - metrics.created_at).total_seconds() > 7200):  # 2 hours

                    self._remove_connection(conn_id)
                    return

        # Return to idle pool
        self._add_to_idle_pool(conn_id)

        # Record metrics
        self.total_queries_executed += 1
        self.total_execution_time += execution_time
        record_metric("connection_pool.query_execution_time", execution_time)
        record_metric("connection_pool.queries_executed", 1)

    def _remove_connection(self, conn_id: str) -> None:
        """Remove connection from pool."""
        with self.lock:
            if conn_id in self.connections:
                with suppress(Exception):
                    self.connections[conn_id].close()

                del self.connections[conn_id]

            if conn_id in self.metrics:
                del self.metrics[conn_id]

            if conn_id in self.active_connections:
                del self.active_connections[conn_id]

        record_metric("connection_pool.connections_removed", 1)
        logger.logger.debug("Connection removed", connection_id=conn_id)

    def health_check(self) -> None:
        """Perform health check on all connections."""
        current_time = time.time()

        if (current_time - self.last_health_check < self.health_check_interval or
            self.health_check_running):
            return

        self.health_check_running = True
        self.last_health_check = current_time

        try:
            broken_connections = []
            idle_too_long = []

            with self.lock:
                for conn_id, metrics in self.metrics.items():
                    # Check for broken connections
                    if metrics.state == ConnectionState.BROKEN:
                        broken_connections.append(conn_id)
                        continue

                    # Check for idle connections that have been idle too long
                    if (metrics.state == ConnectionState.IDLE and
                        (datetime.now() - metrics.last_used).total_seconds() > self.max_idle_time):
                        idle_too_long.append(conn_id)
                        continue

                    # Test active connections periodically
                    if (metrics.state == ConnectionState.IDLE and
                        conn_id in self.connections):
                        conn = self.connections[conn_id]
                        if not self._test_connection(conn_id, conn):
                            broken_connections.append(conn_id)

            # Remove broken and overly idle connections
            for conn_id in broken_connections + idle_too_long:
                self._remove_connection(conn_id)

            # Ensure minimum connections
            current_count = len(self.connections)
            if current_count < self.min_connections:
                for _ in range(self.min_connections - current_count):
                    try:
                        conn_id = self._create_connection()
                        self._add_to_idle_pool(conn_id)
                    except Exception as e:
                        logger.log_error(error=e, context={"operation": "health_check_replenish"})

            # Record health metrics
            record_metric("connection_pool.total_connections", len(self.connections))
            record_metric("connection_pool.idle_connections", self.idle_connections.qsize())
            record_metric("connection_pool.active_connections", len(self.active_connections))

            logger.logger.debug(
                "Health check completed",
                total_connections=len(self.connections),
                broken_removed=len(broken_connections),
                idle_removed=len(idle_too_long),
            )

        finally:
            self.health_check_running = False

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self.lock:
            idle_count = self.idle_connections.qsize()
            active_count = len(self.active_connections)
            total_count = len(self.connections)

            avg_query_time = (
                self.total_execution_time / self.total_queries_executed
                if self.total_queries_executed > 0 else 0
            )

            # Connection age statistics
            now = datetime.now()
            connection_ages = [
                (now - metrics.created_at).total_seconds()
                for metrics in self.metrics.values()
            ]

            avg_connection_age = (
                sum(connection_ages) / len(connection_ages)
                if connection_ages else 0
            )

        return {
            "total_connections": total_count,
            "idle_connections": idle_count,
            "active_connections": active_count,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "utilization_percent": (active_count / total_count * 100) if total_count > 0 else 0,
            "total_connections_created": self.total_connections_created,
            "total_queries_executed": self.total_queries_executed,
            "avg_query_execution_time": avg_query_time,
            "avg_connection_age_seconds": avg_connection_age,
            "health_check_interval": self.health_check_interval,
            "last_health_check": self.last_health_check,
        }

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self.lock:
            for conn_id in list(self.connections.keys()):
                self._remove_connection(conn_id)

            # Clear queues
            while not self.idle_connections.empty():
                try:
                    self.idle_connections.get_nowait()
                except Empty:
                    break

            self.active_connections.clear()

        logger.logger.info("All connections closed")


# Global connection pool instance
_connection_pool: Optional[IntelligentConnectionPool] = None


def initialize_connection_pool(
    database_url: str,
    min_connections: int = 2,
    max_connections: int = 20,
    **kwargs,
) -> None:
    """Initialize the global connection pool."""
    global _connection_pool

    if _connection_pool:
        _connection_pool.close_all()

    _connection_pool = IntelligentConnectionPool(
        database_url=database_url,
        min_connections=min_connections,
        max_connections=max_connections,
        **kwargs,
    )

    logger.logger.info("Global connection pool initialized")


def get_connection_pool() -> Optional[IntelligentConnectionPool]:
    """Get the global connection pool."""
    return _connection_pool


@contextmanager
def get_database_connection():
    """Get a database connection from the global pool."""
    if not _connection_pool:
        msg = "Connection pool not initialized"
        raise RuntimeError(msg)

    with _connection_pool.get_connection() as conn:
        yield conn


def get_pool_statistics() -> dict[str, Any]:
    """Get connection pool statistics."""
    if not _connection_pool:
        return {"error": "Connection pool not initialized"}

    return _connection_pool.get_stats()
