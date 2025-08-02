"""Unit tests for database module.

Tests the DatabaseManager class and related database functionality.
"""

import pytest
from unittest.mock import Mock, patch
from sqlalchemy.exc import SQLAlchemyError

from src.sql_synth.database import DatabaseManager


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

    def test_init_with_valid_db_type(self):
        """Test initialization with valid database type."""
        db_manager = DatabaseManager("sqlite")
        assert db_manager.db_type == "sqlite"
        assert db_manager.connection_string is None

    def test_init_with_invalid_db_type(self):
        """Test initialization with invalid database type raises error."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseManager("invalid_db")

    @patch('src.sql_synth.database.create_engine')
    def test_connect_success(self, mock_create_engine):
        """Test successful database connection."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        db_manager = DatabaseManager("sqlite")
        result = db_manager.connect("sqlite:///test.db")
        
        assert result == mock_engine
        mock_create_engine.assert_called_once()

    @patch('src.sql_synth.database.create_engine')
    def test_connect_failure(self, mock_create_engine):
        """Test database connection failure."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")
        
        db_manager = DatabaseManager("sqlite")
        
        with pytest.raises(SQLAlchemyError):
            db_manager.connect("invalid://connection/string")

    def test_get_dialect_info(self):
        """Test getting dialect information."""
        db_manager = DatabaseManager("postgresql")
        dialect_info = db_manager.get_dialect_info()
        
        assert dialect_info["name"] == "postgresql"
        assert "supports_ilike" in dialect_info
        assert "identifier_quote" in dialect_info

    def test_supports_ilike(self):
        """Test ILIKE support detection."""
        # PostgreSQL supports ILIKE
        pg_manager = DatabaseManager("postgresql")
        assert pg_manager.supports_ilike() is True
        
        # SQLite does not support ILIKE
        sqlite_manager = DatabaseManager("sqlite")
        assert sqlite_manager.supports_ilike() is False

    @pytest.mark.parametrize("db_type,expected_quote", [
        ("sqlite", '"'),
        ("postgresql", '"'),
        ("mysql", "`"),
        ("snowflake", '"'),
    ])
    def test_get_identifier_quote(self, db_type, expected_quote):
        """Test getting correct identifier quote for each database type."""
        db_manager = DatabaseManager(db_type)
        assert db_manager.get_identifier_quote() == expected_quote

    def test_build_connection_string_sqlite(self):
        """Test building SQLite connection string."""
        db_manager = DatabaseManager("sqlite")
        conn_str = db_manager.build_connection_string(
            database="test.db"
        )
        assert conn_str == "sqlite:///test.db"

    def test_build_connection_string_postgresql(self):
        """Test building PostgreSQL connection string."""
        db_manager = DatabaseManager("postgresql")
        conn_str = db_manager.build_connection_string(
            host="localhost",
            port=5432,
            database="testdb",
            username="user",
            password="pass"
        )
        expected = "postgresql://user:pass@localhost:5432/testdb"
        assert conn_str == expected

    def test_build_connection_string_missing_required(self):
        """Test building connection string with missing required parameters."""
        db_manager = DatabaseManager("postgresql")
        
        with pytest.raises(ValueError, match="Missing required parameter"):
            db_manager.build_connection_string(host="localhost")

    @patch.dict(os.environ, {"DB_HOST": "testhost", "DB_PORT": "5432", "DB_NAME": "testdb"})
    def test_from_env_variables(self):
        """Test creating database manager from environment variables."""
        db_manager = DatabaseManager.from_env_variables("postgresql")
        
        assert db_manager.db_type == "postgresql"
        # Additional assertions would depend on the actual implementation

    def test_validate_connection_string(self):
        """Test connection string validation."""
        db_manager = DatabaseManager("sqlite")
        
        # Valid connection string
        assert db_manager.validate_connection_string("sqlite:///test.db") is True
        
        # Invalid connection string
        assert db_manager.validate_connection_string("invalid://string") is False

    @patch('src.sql_synth.database.create_engine')
    def test_test_connection(self, mock_create_engine):
        """Test connection testing functionality."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        db_manager = DatabaseManager("sqlite")
        result = db_manager.test_connection("sqlite:///test.db")
        
        assert result is True
        mock_engine.connect.assert_called_once()

    @patch('src.sql_synth.database.create_engine')
    def test_test_connection_failure(self, mock_create_engine):
        """Test connection testing with failure."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = SQLAlchemyError("Connection failed")
        mock_create_engine.return_value = mock_engine
        
        db_manager = DatabaseManager("sqlite")
        result = db_manager.test_connection("sqlite:///test.db")
        
        assert result is False