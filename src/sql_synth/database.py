"""Database connection and configuration management module.

This module provides a unified interface for connecting to different SQL databases
with support for multiple dialects and secure configuration management.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, ClassVar, Optional

from sqlalchemy import Engine, create_engine, pool
from sqlalchemy.exc import DisconnectionError, SQLAlchemyError, TimeoutError
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and configurations for multiple SQL dialects."""

    SUPPORTED_DB_TYPES: ClassVar[set[str]] = {
        "sqlite", "postgresql", "mysql", "snowflake",
    }

    DIALECT_INFO: ClassVar[dict[str, dict[str, Any]]] = {
        "sqlite": {
            "name": "sqlite",
            "supports_ilike": False,
            "identifier_quote": '"',
            "default_port": None,
        },
        "postgresql": {
            "name": "postgresql",
            "supports_ilike": True,
            "identifier_quote": '"',
            "default_port": 5432,
        },
        "mysql": {
            "name": "mysql",
            "supports_ilike": False,
            "identifier_quote": "`",
            "default_port": 3306,
        },
        "snowflake": {
            "name": "snowflake",
            "supports_ilike": True,
            "identifier_quote": '"',
            "default_port": 443,
        },
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize DatabaseManager with configuration.

        Args:
            config: Dictionary containing database configuration
                   Must include 'db_type' and 'database_url'

        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        self.db_type = config.get("db_type")
        self.database_url = config.get("database_url")
        self.pool_size = config.get("pool_size", 5)
        self.max_overflow = config.get("max_overflow", 10)
        self.pool_timeout = config.get("pool_timeout", 30)
        self.pool_recycle = config.get("pool_recycle", 3600)
        self.connect_timeout = config.get("connect_timeout", 30)

        if not self.db_type:
            msg = "db_type is required in configuration"
            raise ValueError(msg)

        if self.db_type not in self.SUPPORTED_DB_TYPES:
            supported_types = ", ".join(self.SUPPORTED_DB_TYPES)
            msg = (
                f"Unsupported database type: {self.db_type}. "
                f"Supported types: {supported_types}"
            )
            raise ValueError(msg)

        if not self.database_url:
            msg = "database_url is required in configuration"
            raise ValueError(msg)

        self._engine: Optional[Engine] = None
        self._connection_attempts = 0
        self._max_connection_attempts = 3
        self._connection_retry_delay = 2

        logger.info("DatabaseManager initialized for %s", self.db_type)

    def get_engine(self) -> Engine:
        """Get SQLAlchemy engine for database connection with retry logic.

        Returns:
            SQLAlchemy Engine instance

        Raises:
            SQLAlchemyError: If engine creation fails after retries
        """
        if self._engine is None:
            if self.database_url is None:
                msg = "Database URL not configured"
                raise ValueError(msg)

            for attempt in range(1, self._max_connection_attempts + 1):
                try:
                    # Configure engine with connection pooling and timeouts
                    engine_kwargs = {
                        "pool_pre_ping": True,  # Validate connections before use
                        "connect_args": {"connect_timeout": self.connect_timeout},
                    }

                    # Adjust for database-specific configurations
                    if self.db_type == "sqlite":
                        engine_kwargs.update({
                            "poolclass": pool.StaticPool,
                            "connect_args": {"check_same_thread": False, "timeout": self.connect_timeout},
                        })
                    else:
                        # Only add pool parameters for non-SQLite databases
                        engine_kwargs.update({
                            "pool_size": self.pool_size,
                            "max_overflow": self.max_overflow,
                            "pool_timeout": self.pool_timeout,
                            "pool_recycle": self.pool_recycle,
                        })

                    if self.db_type == "postgresql":
                        engine_kwargs["connect_args"].update({
                            "connect_timeout": self.connect_timeout,
                            "application_name": "sql-synth-agent",
                        })
                    elif self.db_type == "mysql":
                        engine_kwargs["connect_args"].update({
                            "connect_timeout": self.connect_timeout,
                            "charset": "utf8mb4",
                        })

                    self._engine = create_engine(self.database_url, **engine_kwargs)

                    # Test the engine with a simple connection
                    with self._engine.connect() as conn:
                        conn.execute(text("SELECT 1"))

                    logger.info("Created engine for %s database (attempt %d)", self.db_type, attempt)
                    break

                except Exception as e:
                    logger.warning("Engine creation attempt %d failed: %s", attempt, str(e))
                    if attempt == self._max_connection_attempts:
                        logger.exception("Failed to create engine after %d attempts", self._max_connection_attempts)
                        msg = f"Engine creation failed after {self._max_connection_attempts} attempts: {e}"
                        raise SQLAlchemyError(msg) from e
                    time.sleep(self._connection_retry_delay)

        return self._engine

    def test_connection(self) -> dict[str, Any]:
        """Test database connection with detailed diagnostics.

        Returns:
            Dictionary with connection test results and diagnostics
        """
        start_time = time.time()

        try:
            engine = self.get_engine()
            with engine.connect() as connection:
                # Execute a simple query to test the connection
                result = connection.execute(text("SELECT 1 as test_value"))
                test_value = result.fetchone()[0]

                # Get connection info
                dialect_name = connection.dialect.name
                server_version = getattr(connection.dialect, "server_version_info", None)

                connection_time = time.time() - start_time

                logger.info("Database connection test successful in %.3fs", connection_time)

                return {
                    "success": True,
                    "connection_time": connection_time,
                    "dialect": dialect_name,
                    "server_version": str(server_version) if server_version else "Unknown",
                    "test_query_result": test_value,
                    "pool_info": {
                        "pool_size": self.pool_size,
                        "max_overflow": self.max_overflow,
                        "checked_in": getattr(engine.pool, "checkedin", lambda: 0)(),
                        "checked_out": getattr(engine.pool, "checkedout", lambda: 0)(),
                        "overflow": getattr(engine.pool, "overflow", lambda: 0)(),
                    } if hasattr(engine, "pool") else None,
                }

        except SQLAlchemyError as e:
            connection_time = time.time() - start_time
            logger.exception("Database connection test failed after %.3fs", connection_time)
            return {
                "success": False,
                "error": str(e),
                "error_type": "SQLAlchemyError",
                "connection_time": connection_time,
            }
        except Exception as e:
            connection_time = time.time() - start_time
            logger.exception("Unexpected error during connection test after %.3fs", connection_time)
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnexpectedError",
                "connection_time": connection_time,
            }

    def get_dialect_info(self) -> dict[str, Any]:
        """Get dialect-specific information.

        Returns:
            Dictionary containing dialect information
        """
        if self.db_type is None:
            msg = "Database type not configured"
            raise ValueError(msg)
        return self.DIALECT_INFO[self.db_type].copy()

    @contextmanager
    def get_connection(self, autocommit: bool = True):
        """Context manager for database connections with automatic cleanup.

        Args:
            autocommit: Whether to automatically commit transactions

        Yields:
            SQLAlchemy Connection object

        Raises:
            SQLAlchemyError: If connection fails
        """
        engine = self.get_engine()
        connection = None
        transaction = None

        try:
            connection = engine.connect()
            if not autocommit:
                transaction = connection.begin()

            yield connection

            if transaction:
                transaction.commit()

        except Exception as e:
            if transaction:
                try:
                    transaction.rollback()
                    logger.info("Transaction rolled back due to error")
                except Exception as rollback_error:
                    logger.exception("Failed to rollback transaction: %s", rollback_error)

            logger.exception("Database operation failed: %s", str(e))
            raise

        finally:
            if connection:
                try:
                    connection.close()
                except Exception as close_error:
                    logger.warning("Error closing connection: %s", close_error)

    def execute_query_safely(self, query: str, parameters: Optional[dict] = None, limit: Optional[int] = None) -> dict[str, Any]:
        """Execute a query safely with error handling and limits.

        Args:
            query: SQL query to execute
            parameters: Query parameters for safe execution
            limit: Maximum number of rows to return

        Returns:
            Dictionary with query results and metadata
        """
        start_time = time.time()

        try:
            # Add LIMIT if specified and not already present
            if limit and "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"

            with self.get_connection() as connection:
                # Use parameterized query for security
                if parameters:
                    result = connection.execute(text(query), parameters)
                else:
                    result = connection.execute(text(query))

                execution_time = time.time() - start_time

                if result.returns_rows:
                    rows = result.fetchall()
                    columns = list(result.keys())

                    return {
                        "success": True,
                        "rows": [dict(zip(columns, row)) for row in rows],
                        "columns": columns,
                        "row_count": len(rows),
                        "execution_time": execution_time,
                        "query": query,
                        "parameters": parameters,
                    }
                return {
                    "success": True,
                    "message": "Query executed successfully (no rows returned)",
                    "execution_time": execution_time,
                    "query": query,
                    "parameters": parameters,
                }

        except (SQLAlchemyError, DisconnectionError, TimeoutError) as e:
            execution_time = time.time() - start_time
            logger.exception("SQL execution failed after %.3fs: %s", execution_time, str(e))

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time,
                "query": query,
                "parameters": parameters,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception("Unexpected error during query execution after %.3fs: %s", execution_time, str(e))

            return {
                "success": False,
                "error": str(e),
                "error_type": "UnexpectedError",
                "execution_time": execution_time,
                "query": query,
                "parameters": parameters,
            }

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Dictionary with connection pool information
        """
        if not self._engine or not hasattr(self._engine, "pool"):
            return {"message": "No connection pool available"}

        pool = self._engine.pool

        return {
            "pool_size": getattr(pool, "size", lambda: 0)(),
            "checked_in": getattr(pool, "checkedin", lambda: 0)(),
            "checked_out": getattr(pool, "checkedout", lambda: 0)(),
            "overflow": getattr(pool, "overflow", lambda: 0)(),
            "invalid": getattr(pool, "invalid", lambda: 0)(),
            "pool_timeout": self.pool_timeout,
            "max_overflow": self.max_overflow,
            "pool_recycle": self.pool_recycle,
        }

    def close(self) -> None:
        """Close database engine and connections safely."""
        if self._engine:
            try:
                # Close all connections in the pool
                self._engine.dispose()
                logger.info("Database engine disposed successfully")
            except Exception as e:
                logger.exception("Error disposing database engine: %s", str(e))
            finally:
                self._engine = None


def get_database_manager() -> DatabaseManager:
    """Factory function to create DatabaseManager from environment variables.

    Returns:
        Configured DatabaseManager instance

    Raises:
        ValueError: If required environment variables are missing
    """
    # Try different environment variable names for database type
    db_type = os.getenv("DB_TYPE") or os.getenv("DATABASE_TYPE")
    if not db_type:
        msg = "DB_TYPE or DATABASE_TYPE environment variable is required"
        raise ValueError(msg)

    # Handle Snowflake special case
    database_url: str
    if db_type == "snowflake":
        database_url = _build_snowflake_url()
    else:
        database_url_env = os.getenv("DATABASE_URL")
        if not database_url_env:
            msg = "DATABASE_URL environment variable is required"
            raise ValueError(msg)
        database_url = database_url_env

    # Get connection pool configuration from environment
    config = {
        "db_type": db_type,
        "database_url": database_url,
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
    }

    return DatabaseManager(config)


def _build_snowflake_url() -> str:
    """Build Snowflake connection URL from environment variables.

    Returns:
        Snowflake connection URL string

    Raises:
        ValueError: If required Snowflake environment variables are missing
    """
    required_vars = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA", "SNOWFLAKE_WAREHOUSE",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        missing_vars_str = ", ".join(missing_vars)
        msg = f"Missing required Snowflake environment variables: {missing_vars_str}"
        raise ValueError(msg)

    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    return f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"
