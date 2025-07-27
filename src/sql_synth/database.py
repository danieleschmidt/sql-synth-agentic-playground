"""Database connection and configuration management module.

This module provides a unified interface for connecting to different SQL databases
with support for multiple dialects and secure configuration management.
"""

import logging
import os
from typing import Any, ClassVar, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError
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

        logger.info("DatabaseManager initialized for %s", self.db_type)

    def get_engine(self) -> Engine:
        """Get SQLAlchemy engine for database connection.

        Returns:
            SQLAlchemy Engine instance
        """
        if self._engine is None:
            assert self.database_url is not None  # Checked in __init__
            self._engine = create_engine(self.database_url)
            logger.info("Created engine for %s database", self.db_type)

        return self._engine

    def test_connection(self) -> bool:
        """Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            engine = self.get_engine()
            with engine.connect() as connection:
                # Execute a simple query to test the connection
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            logger.exception("Database connection test failed")
            return False
        except Exception:
            logger.exception("Unexpected error during connection test")
            return False
        else:
            logger.info("Database connection test successful")
            return True

    def get_dialect_info(self) -> dict[str, Any]:
        """Get dialect-specific information.

        Returns:
            Dictionary containing dialect information
        """
        assert self.db_type is not None  # Checked in __init__
        return self.DIALECT_INFO[self.db_type].copy()

    def close(self) -> None:
        """Close database engine and connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database engine closed")


def get_database_manager() -> DatabaseManager:
    """Factory function to create DatabaseManager from environment variables.

    Returns:
        Configured DatabaseManager instance

    Raises:
        ValueError: If required environment variables are missing
    """
    db_type = os.getenv("DB_TYPE")
    if not db_type:
        msg = "DB_TYPE environment variable is required"
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

    config = {
        "db_type": db_type,
        "database_url": database_url,
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
