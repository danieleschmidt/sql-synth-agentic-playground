"""Tests for database connection and configuration module."""

import os
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine

from src.sql_synth.database import DatabaseManager, get_database_manager


class TestDatabaseManager:
    """Test DatabaseManager class."""

    def test_init_with_valid_config(self):
        """Test DatabaseManager initialization with valid configuration."""
        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db",
        }
        db_manager = DatabaseManager(config)
        assert db_manager.db_type == "sqlite"
        assert db_manager.database_url == "sqlite:///test.db"

    def test_init_with_invalid_db_type(self):
        """Test DatabaseManager initialization with invalid db_type."""
        config = {
            "db_type": "invalid_db",
            "database_url": "sqlite:///test.db",
        }
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseManager(config)

    def test_init_with_missing_config(self):
        """Test DatabaseManager initialization with missing configuration."""
        config = {"db_type": "sqlite"}
        with pytest.raises(ValueError, match="database_url is required"):
            DatabaseManager(config)

    @patch("src.sql_synth.database.create_engine")
    def test_get_engine_sqlite(self, mock_create_engine):
        """Test get_engine for SQLite."""
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db",
        }
        db_manager = DatabaseManager(config)
        engine = db_manager.get_engine()

        assert engine == mock_engine
        mock_create_engine.assert_called_once_with("sqlite:///test.db")

    @patch("src.sql_synth.database.create_engine")
    def test_get_engine_postgresql(self, mock_create_engine):
        """Test get_engine for PostgreSQL."""
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        config = {
            "db_type": "postgresql",
            "database_url": "postgresql://user:pass@localhost:5432/db",
        }
        db_manager = DatabaseManager(config)
        engine = db_manager.get_engine()

        assert engine == mock_engine
        mock_create_engine.assert_called_once_with("postgresql://user:pass@localhost:5432/db")

    @patch("src.sql_synth.database.create_engine")
    def test_test_connection_success(self, mock_create_engine):
        """Test successful connection test."""
        mock_engine = Mock(spec=Engine)
        mock_connection = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_connection)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_engine.connect.return_value = mock_context_manager
        mock_create_engine.return_value = mock_engine

        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db",
        }
        db_manager = DatabaseManager(config)

        result = db_manager.test_connection()
        assert result is True

    @patch("src.sql_synth.database.create_engine")
    def test_test_connection_failure(self, mock_create_engine):
        """Test connection test failure."""
        mock_engine = Mock(spec=Engine)
        mock_engine.connect.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine

        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db",
        }
        db_manager = DatabaseManager(config)

        result = db_manager.test_connection()
        assert result is False

    def test_get_dialect_info(self):
        """Test get_dialect_info method."""
        config = {
            "db_type": "postgresql",
            "database_url": "postgresql://user:pass@localhost:5432/db",
        }
        db_manager = DatabaseManager(config)
        dialect_info = db_manager.get_dialect_info()

        assert dialect_info["name"] == "postgresql"
        assert dialect_info["supports_ilike"] is True
        assert dialect_info["identifier_quote"] == '"'


class TestGetDatabaseManager:
    """Test get_database_manager factory function."""

    @patch.dict(os.environ, {
        "DB_TYPE": "sqlite",
        "DATABASE_URL": "sqlite:///test.db",
    })
    def test_get_database_manager_from_env(self):
        """Test creating DatabaseManager from environment variables."""
        db_manager = get_database_manager()
        assert db_manager.db_type == "sqlite"
        assert db_manager.database_url == "sqlite:///test.db"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_database_manager_missing_env(self):
        """Test creating DatabaseManager with missing environment variables."""
        with pytest.raises(
            ValueError, match="DB_TYPE environment variable is required",
        ):
            get_database_manager()

    @patch.dict(os.environ, {
        "DB_TYPE": "snowflake",
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_USER": "test_user",
        "SNOWFLAKE_PASSWORD": "test_pass",
        "SNOWFLAKE_DATABASE": "test_db",
        "SNOWFLAKE_SCHEMA": "test_schema",
        "SNOWFLAKE_WAREHOUSE": "test_wh",
    })
    def test_get_database_manager_snowflake_from_env(self):
        """Test creating DatabaseManager for Snowflake from environment variables."""
        db_manager = get_database_manager()
        assert db_manager.db_type == "snowflake"
        expected_url = "snowflake://test_user:test_pass@test_account/test_db/test_schema?warehouse=test_wh"
        assert db_manager.database_url == expected_url

