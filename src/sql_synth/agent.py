"""SQL Synthesis Agent using LangChain.

This module provides the core natural language to SQL translation functionality
using LangChain's SQL agent toolkit with comprehensive security and validation.
"""

import logging
import time
from typing import Any, ClassVar, Optional

from langchain.agents import AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from .database import DatabaseManager
from .security import security_auditor
from .metrics import QueryMetrics
from .cache import cache_generation_result, cache_query_result, cache_manager
from .concurrent import concurrent_task

logger = logging.getLogger(__name__)


class SQLSynthesisAgent:
    """Natural language to SQL translation agent with security and validation."""

    MAX_QUERY_LENGTH: ClassVar[int] = 10000
    MAX_EXECUTION_TIME: ClassVar[int] = 30  # seconds
    SUPPORTED_OPERATIONS: ClassVar[set[str]] = {
        "SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW"
    }

    def __init__(
        self,
        database_manager: DatabaseManager,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the SQL synthesis agent.

        Args:
            database_manager: Database connection manager
            model_name: LLM model to use for SQL generation
            temperature: Model temperature for generation
            max_retries: Maximum number of retry attempts

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If agent initialization fails
        """
        self.database_manager = database_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Use global security auditor
        self.security_validator = security_auditor
        
        # Initialize metrics tracker
        self.metrics = QueryMetrics()
        
        # Create LangChain SQL database instance
        try:
            engine = database_manager.get_engine()
            self.sql_database = SQLDatabase(engine=engine)
            logger.info("SQLDatabase instance created successfully")
        except Exception as e:
            logger.exception("Failed to create SQLDatabase instance")
            raise RuntimeError(f"SQLDatabase creation failed: {e}") from e
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                request_timeout=self.MAX_EXECUTION_TIME,
            )
            logger.info("LLM initialized: %s", model_name)
        except Exception as e:
            logger.exception("Failed to initialize LLM")
            raise RuntimeError(f"LLM initialization failed: {e}") from e
        
        # Create SQL agent
        try:
            toolkit = SQLDatabaseToolkit(db=self.sql_database, llm=self.llm)
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_execution_time=self.MAX_EXECUTION_TIME,
                max_iterations=3,
            )
            logger.info("SQL agent created successfully")
        except Exception as e:
            logger.exception("Failed to create SQL agent")
            raise RuntimeError(f"SQL agent creation failed: {e}") from e

    @cache_generation_result(ttl=3600)
    @concurrent_task(timeout=30)
    def generate_sql(self, natural_language_query: str) -> dict[str, Any]:
        """Generate SQL from natural language query.

        Args:
            natural_language_query: User's natural language input

        Returns:
            Dictionary containing SQL query, metadata, and execution info

        Raises:
            ValueError: If input validation fails
            RuntimeError: If SQL generation fails
        """
        start_time = time.time()
        
        try:
            # Input validation
            self._validate_input(natural_language_query)
            
            # Generate SQL using LangChain agent
            result = self._generate_with_retry(natural_language_query)
            
            # Extract SQL from agent response
            sql_query = self._extract_sql_from_result(result)
            
            # Security validation
            is_safe, violations = self.security_validator.audit_generated_query(sql_query)
            if not is_safe:
                violation_reasons = [str(v) for v in violations]
                raise ValueError(f"Security validation failed: {'; '.join(violation_reasons)}")
            
            # Additional validations
            self._validate_generated_sql(sql_query)
            
            generation_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_generation(
                success=True,
                generation_time=generation_time,
                query_length=len(sql_query),
            )
            
            return {
                "sql_query": sql_query,
                "success": True,
                "generation_time": generation_time,
                "security_validated": True,
                "metadata": {
                    "model_used": self.model_name,
                    "original_query": natural_language_query,
                    "query_length": len(sql_query),
                    "dialect": self.database_manager.db_type,
                    "timestamp": time.time(),
                },
                "agent_output": result,
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            
            # Record failed metrics
            self.metrics.record_generation(
                success=False,
                generation_time=generation_time,
                error=str(e),
            )
            
            logger.exception("SQL generation failed for query: %s", natural_language_query[:100])
            
            return {
                "sql_query": None,
                "success": False,
                "error": str(e),
                "generation_time": generation_time,
                "metadata": {
                    "model_used": self.model_name,
                    "original_query": natural_language_query,
                    "timestamp": time.time(),
                },
            }

    @cache_query_result(ttl=1800)
    def execute_sql(self, sql_query: str, limit: int = 100) -> dict[str, Any]:
        """Execute SQL query with safety checks.

        Args:
            sql_query: SQL query to execute
            limit: Maximum number of rows to return

        Returns:
            Dictionary containing execution results and metadata

        Raises:
            ValueError: If query validation fails
            RuntimeError: If execution fails
        """
        start_time = time.time()
        
        try:
            # Security validation
            is_safe, violations = self.security_validator.audit_generated_query(sql_query)
            if not is_safe:
                violation_reasons = [str(v) for v in violations]
                raise ValueError(f"Security validation failed: {'; '.join(violation_reasons)}")
            
            # Add LIMIT clause if not present and not a metadata query
            limited_query = self._add_limit_if_needed(sql_query, limit)
            
            # Execute query
            engine = self.database_manager.get_engine()
            with engine.connect() as connection:
                result = connection.execute(text(limited_query))
                
                # Fetch results
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = list(result.keys())
                    
                    execution_time = time.time() - start_time
                    
                    # Record successful execution metrics
                    self.metrics.record_execution(
                        success=True,
                        execution_time=execution_time,
                        rows_returned=len(rows),
                    )
                    
                    return {
                        "success": True,
                        "rows": [dict(zip(columns, row)) for row in rows],
                        "columns": columns,
                        "row_count": len(rows),
                        "execution_time": execution_time,
                        "query_executed": limited_query,
                    }
                else:
                    # Non-SELECT query (shouldn't happen with current restrictions)
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "message": "Query executed successfully (no results returned)",
                        "execution_time": execution_time,
                        "query_executed": limited_query,
                    }
                    
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed execution metrics
            self.metrics.record_execution(
                success=False,
                execution_time=execution_time,
                error=str(e),
            )
            
            logger.exception("SQL execution failed: %s", sql_query[:100])
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "query_executed": sql_query,
            }

    def get_schema_info(self) -> dict[str, Any]:
        """Get database schema information.

        Returns:
            Dictionary containing schema information
        """
        try:
            return {
                "tables": self.sql_database.get_usable_table_names(),
                "dialect": self.database_manager.get_dialect_info(),
                "database_type": self.database_manager.db_type,
            }
        except Exception as e:
            logger.exception("Failed to get schema info")
            return {"error": str(e)}

    def get_metrics(self) -> dict[str, Any]:
        """Get agent performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        return self.metrics.get_summary()

    def _validate_input(self, query: str) -> None:
        """Validate natural language input.

        Args:
            query: Natural language query to validate

        Raises:
            ValueError: If validation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (max {self.MAX_QUERY_LENGTH} characters)")
        
        # Check for potential SQL injection in natural language
        suspicious_patterns = ["';", "--", "/*", "*/", "xp_", "sp_"]
        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if pattern in query_lower:
                logger.warning("Suspicious pattern detected in input: %s", pattern)

    def _generate_with_retry(self, query: str) -> str:
        """Generate SQL with retry logic.

        Args:
            query: Natural language query

        Returns:
            Agent response string

        Raises:
            RuntimeError: If all retries fail
        """
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("SQL generation attempt %d/%d", attempt, self.max_retries)
                result = self.agent.run(query)
                return result
            except Exception as e:
                last_error = e
                logger.warning("Attempt %d failed: %s", attempt, str(e))
                if attempt < self.max_retries:
                    time.sleep(1)  # Brief delay before retry
        
        raise RuntimeError(f"SQL generation failed after {self.max_retries} attempts: {last_error}")

    def _extract_sql_from_result(self, result: str) -> str:
        """Extract SQL query from agent result.

        Args:
            result: Agent response containing SQL

        Returns:
            Cleaned SQL query string

        Raises:
            ValueError: If no valid SQL found
        """
        # The agent typically returns explanatory text with SQL
        # We need to extract just the SQL portion
        lines = result.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Look for SQL keywords to identify SQL content
            if any(keyword in line_stripped.upper() for keyword in ['SELECT', 'WITH', 'EXPLAIN']):
                in_sql = True
                sql_lines.append(line_stripped)
            elif in_sql and line_stripped:
                # Continue collecting SQL lines
                if line_stripped.endswith(';'):
                    sql_lines.append(line_stripped)
                    break
                sql_lines.append(line_stripped)
            elif in_sql and not line_stripped:
                # Empty line might end SQL block
                break
        
        if not sql_lines:
            # Fallback: look for any SQL-like content
            for line in lines:
                if 'SELECT' in line.upper():
                    return line.strip()
            raise ValueError("No SQL query found in agent response")
        
        sql_query = ' '.join(sql_lines)
        
        # Clean up the SQL
        sql_query = sql_query.strip()
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query

    def _validate_generated_sql(self, sql_query: str) -> None:
        """Validate generated SQL query.

        Args:
            sql_query: SQL query to validate

        Raises:
            ValueError: If validation fails
        """
        if not sql_query or not sql_query.strip():
            raise ValueError("Generated SQL is empty")
        
        sql_upper = sql_query.upper().strip()
        
        # Check if query starts with allowed operations
        allowed_start = any(sql_upper.startswith(op) for op in self.SUPPORTED_OPERATIONS)
        if not allowed_start:
            operations_str = ", ".join(self.SUPPORTED_OPERATIONS)
            raise ValueError(f"Query must start with one of: {operations_str}")
        
        # Check for dangerous operations
        dangerous_keywords = [
            "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", 
            "TRUNCATE", "EXEC", "EXECUTE", "CALL"
        ]
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                raise ValueError(f"Dangerous operation not allowed: {keyword}")

    def _add_limit_if_needed(self, sql_query: str, limit: int) -> str:
        """Add LIMIT clause to query if not present.

        Args:
            sql_query: Original SQL query
            limit: Maximum number of rows

        Returns:
            SQL query with LIMIT clause if needed
        """
        sql_upper = sql_query.upper()
        
        # Skip LIMIT for certain query types
        if any(keyword in sql_upper for keyword in ["EXPLAIN", "DESCRIBE", "SHOW"]):
            return sql_query
        
        # Check if LIMIT already exists
        if "LIMIT" in sql_upper:
            return sql_query
        
        # Add LIMIT clause
        if sql_query.rstrip().endswith(';'):
            sql_query = sql_query.rstrip()[:-1]  # Remove semicolon
        
        return f"{sql_query} LIMIT {limit};"


class AgentFactory:
    """Factory for creating SQL synthesis agents."""

    @staticmethod
    def create_agent(
        database_manager: DatabaseManager,
        model_name: str = "gpt-3.5-turbo",
        **kwargs: Any,
    ) -> SQLSynthesisAgent:
        """Create a SQL synthesis agent.

        Args:
            database_manager: Database connection manager
            model_name: LLM model to use
            **kwargs: Additional agent configuration

        Returns:
            Configured SQLSynthesisAgent instance

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If agent creation fails
        """
        try:
            return SQLSynthesisAgent(
                database_manager=database_manager,
                model_name=model_name,
                **kwargs,
            )
        except Exception as e:
            logger.exception("Failed to create SQL synthesis agent")
            raise RuntimeError(f"Agent creation failed: {e}") from e