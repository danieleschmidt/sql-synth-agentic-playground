"""Unit tests for database module.

Tests the DatabaseManager class and related database functionality.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from src.sql_synth.database import DatabaseManager, get_database_manager


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""

    def test_supported_db_types(self):
        """Test that supported database types are correctly defined."""
        expected_types = {"sqlite", "postgresql", "mysql", "snowflake"}
        assert DatabaseManager.SUPPORTED_DB_TYPES == expected_types

    def test_dialect_info_structure(self):
        """Test that dialect info contains required keys."""
        required_keys = {"name", "supports_ilike", "identifier_quote"}
        
        for dialect, info in DatabaseManager.DIALECT_INFO.items():
            assert required_keys.issubset(info.keys())
            assert isinstance(info["supports_ilike"], bool)
            assert isinstance(info["identifier_quote"], str)

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db"
        }
        db_manager = DatabaseManager(config)
        assert db_manager.db_type == "sqlite"
        assert db_manager.database_url == "sqlite:///test.db"

    def test_init_with_invalid_db_type(self):
        """Test initialization with invalid database type raises error."""
        config = {
            "db_type": "invalid_db",
            "database_url": "invalid://url"
        }
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseManager(config)

    def test_init_missing_db_type(self):
        """Test initialization with missing db_type raises error."""
        config = {
            "database_url": "sqlite:///test.db"
        }
        with pytest.raises(ValueError, match="db_type is required"):
            DatabaseManager(config)

    def test_init_missing_database_url(self):
        """Test initialization with missing database_url raises error."""
        config = {
            "db_type": "sqlite"
        }
        with pytest.raises(ValueError, match="database_url is required"):
            DatabaseManager(config)

    @patch('src.sql_synth.database.create_engine')
    def test_get_engine_success(self, mock_create_engine):
        """Test successful engine creation."""
        mock_engine = Mock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine
        
        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db"
        }
        db_manager = DatabaseManager(config)
        result = db_manager.get_engine()
        
        assert result == mock_engine

    def test_get_dialect_info(self):
        """Test getting dialect information."""
        config = {
            "db_type": "postgresql",
            "database_url": "postgresql://user:pass@localhost/test"
        }
        db_manager = DatabaseManager(config)
        dialect_info = db_manager.get_dialect_info()
        
        assert dialect_info["name"] == "postgresql"
        assert "supports_ilike" in dialect_info
        assert "identifier_quote" in dialect_info

    @patch('src.sql_synth.database.create_engine')
    def test_test_connection_success(self, mock_create_engine):
        """Test successful connection test."""
        mock_engine = Mock()
        mock_connection = MagicMock()
        mock_result = Mock()
        mock_result.fetchone.return_value = [1]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine
        
        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db"
        }
        db_manager = DatabaseManager(config)
        result = db_manager.test_connection()
        
        assert result["success"] is True
        assert "connection_time" in result

    @patch('src.sql_synth.database.create_engine')
    def test_test_connection_failure(self, mock_create_engine):
        """Test connection test failure."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = SQLAlchemyError("Connection failed")
        mock_create_engine.return_value = mock_engine
        
        config = {
            "db_type": "sqlite",
            "database_url": "sqlite:///test.db"
        }
        db_manager = DatabaseManager(config)
        result = db_manager.test_connection()
        
        assert result["success"] is False
        assert "error" in result

    @patch.dict(os.environ, {
        "DATABASE_TYPE": "sqlite",
        "DATABASE_URL": "sqlite:///test.db"
    }, clear=True)
    def test_get_database_manager_from_env(self):
        """Test creating database manager from environment variables."""
        db_manager = get_database_manager()
        
        assert db_manager.db_type == "sqlite"
        assert db_manager.database_url == "sqlite:///test.db"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_database_manager_missing_env(self):
        """Test error when environment variables are missing."""
        with pytest.raises(ValueError, match="DATABASE_TYPE environment variable"):
            get_database_manager()