"""SQL Synthesis Agent using LangChain.

This module provides the core natural language to SQL translation functionality
using LangChain's SQL agent toolkit with comprehensive security and validation.
"""

import logging
import time
from typing import Any, ClassVar

from langchain.agents import AgentType
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from .advanced_validation import ValidationSeverity, global_validator
from .cache import cache_generation_result, cache_query_result
from .concurrent_execution import concurrent_task
from .database import DatabaseManager
from .error_handling import (
    error_context,
    global_error_manager,
)
from .intelligent_performance_engine import global_performance_engine
from .metrics import QueryMetrics
from .performance_optimizer import (
    global_profiler,
    global_query_optimizer,
    optimize_operation,
)
from .security import security_auditor
from .quantum_transcendent_enhancement_engine import execute_quantum_transcendent_enhancement, OptimizationDimension
from .transcendent_sql_optimizer import optimize_sql_transcendent
from .transcendent_error_resilience_framework import handle_transcendent_error, TranscendentErrorContext
from .infinite_scale_performance_nexus import execute_with_infinite_scaling, start_infinite_scaling_systems

logger = logging.getLogger(__name__)


class SQLSynthesisAgent:
    """Natural language to SQL translation agent with security and validation."""

    MAX_QUERY_LENGTH: ClassVar[int] = 10000
    MAX_EXECUTION_TIME: ClassVar[int] = 30  # seconds
    SUPPORTED_OPERATIONS: ClassVar[set[str]] = {
        "SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW",
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

        # Start intelligent performance engine
        global_performance_engine.start_intelligent_optimization()
        
        # Start infinite scaling systems
        asyncio.create_task(start_infinite_scaling_systems())

        # Create LangChain SQL database instance
        try:
            engine = database_manager.get_engine()
            self.sql_database = SQLDatabase(engine=engine)
            logger.info("SQLDatabase instance created successfully")
        except Exception as e:
            logger.exception("Failed to create SQLDatabase instance")
            msg = f"SQLDatabase creation failed: {e}"
            raise RuntimeError(msg) from e

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
            msg = f"LLM initialization failed: {e}"
            raise RuntimeError(msg) from e

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
            msg = f"SQL agent creation failed: {e}"
            raise RuntimeError(msg) from e

    @optimize_operation("sql_generation")
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
            # Use transcendent error handling context
            async with TranscendentErrorContext(
                "sql_generation_transcendent",
                enable_quantum=True,
                enable_consciousness=True,
                enable_autonomous=True
            ):
                with error_context(
                    "sql_generation",
                    global_error_manager,
                    {"query_length": len(natural_language_query), "model": self.model_name},
                ):
                # Advanced input validation
                validation_result = global_validator.validate_natural_language(natural_language_query)
                if not validation_result.is_valid:
                    error_issues = [issue for issue in validation_result.issues
                                  if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
                    error_messages = [issue.message for issue in error_issues]
                    msg = f"Input validation failed: {'; '.join(error_messages)}"
                    raise ValueError(msg)

                # Log validation warnings
                warning_issues = [issue for issue in validation_result.issues
                                if issue.severity == ValidationSeverity.WARNING]
                if warning_issues:
                    warning_messages = [issue.message for issue in warning_issues]
                    logger.warning(f"Input validation warnings: {'; '.join(warning_messages)}")

                # Basic input validation (legacy)
                self._validate_input(natural_language_query)

                # Generate SQL using LangChain agent with infinite scaling
                result = await execute_with_infinite_scaling(
                    self._generate_with_retry,
                    natural_language_query,
                    operation_id=f"sql_generation_{hash(natural_language_query) % 10000}",
                    consciousness_context=validation_result.validation_score * 0.8,
                    enable_quantum_parallel=True,
                    enable_transcendent_caching=True
                )

                # Extract SQL from agent response
                sql_query = self._extract_sql_from_result(result)

                # Apply quantum transcendent enhancement
                quantum_enhancement_result = await execute_quantum_transcendent_enhancement(
                    natural_language_query,
                    [OptimizationDimension.PERFORMANCE, OptimizationDimension.TRANSCENDENCE, OptimizationDimension.INTELLIGENCE]
                )
                
                logger.info(f"🌟 Quantum transcendent enhancement - Level: {quantum_enhancement_result.transcendence_level:.3f}")
                
                # Apply transcendent SQL optimization
                transcendent_optimization = await optimize_sql_transcendent(
                    sql_query,
                    enable_quantum_enhancement=True,
                    enable_consciousness_integration=True
                )
                
                if transcendent_optimization.optimization_score > 0.8:
                    logger.info(f"⚡ Transcendent SQL optimization applied - Score: {transcendent_optimization.optimization_score:.3f}")
                    sql_query = transcendent_optimization.optimized_query

                # Apply intelligent performance optimization (legacy fallback)
                intelligent_optimization = global_performance_engine.optimize_query_execution(
                    sql_query,
                    {
                        "natural_query": natural_language_query[:100],
                        "complexity_threshold": 5.0,
                        "performance_target": 1.0,
                    },
                )

                if intelligent_optimization.get("optimizations_applied"):
                    logger.info(f"Applied intelligent optimizations: {intelligent_optimization['optimizations_applied']}")
                    sql_query = intelligent_optimization["optimized_query"]

                # Traditional optimization as fallback
                optimization_result = global_query_optimizer.optimize_query(
                    sql_query,
                    {
                        "default_limit": 100,
                        "context": "sql_synthesis",
                        "natural_query": natural_language_query[:100],
                    },
                )

                if optimization_result["optimizations_applied"]:
                    logger.info(f"Applied SQL optimizations: {optimization_result['optimizations_applied']}")
                    sql_query = optimization_result["optimized_query"]

                # Security validation
                is_safe, violations = self.security_validator.audit_generated_query(sql_query)
                if not is_safe:
                    violation_reasons = [str(v) for v in violations]
                    msg = f"Security validation failed: {'; '.join(violation_reasons)}"
                    raise ValueError(msg)

                # Advanced SQL validation
                schema_info = self.get_schema_info()
                sql_validation_result = global_validator.validate_sql(sql_query, schema_info)

                if not sql_validation_result.is_valid:
                    error_issues = [issue for issue in sql_validation_result.issues
                                  if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
                    error_messages = [issue.message for issue in error_issues]
                    msg = f"SQL validation failed: {'; '.join(error_messages)}"
                    raise ValueError(msg)

                # Log SQL validation warnings
                warning_issues = [issue for issue in sql_validation_result.issues
                                if issue.severity == ValidationSeverity.WARNING]
                if warning_issues:
                    warning_messages = [issue.message for issue in warning_issues]
                    logger.warning(f"SQL validation warnings: {'; '.join(warning_messages)}")

                # Legacy validation (kept for compatibility)
                self._validate_generated_sql(sql_query)

                generation_time = time.time() - start_time

                # Record metrics for traditional system
                self.metrics.record_generation(
                    success=True,
                    generation_time=generation_time,
                    query_length=len(sql_query),
                )

                # Record performance metrics for intelligent engine
                global_performance_engine.record_performance({
                    "response_time": generation_time,
                    "query_complexity": intelligent_optimization.get("complexity_score", 0.0),
                    "optimization_score": intelligent_optimization.get("optimization_score", 0.0),
                    "cache_hit_rate": 0.0,  # Will be updated by caching system
                    "error_rate": 0.0,
                    "concurrent_requests": 1,
                })

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
                        "validation_score": sql_validation_result.validation_score,
                        "query_characteristics": sql_validation_result.metadata.get("characteristics", {}),
                        "estimated_complexity": sql_validation_result.metadata.get("characteristics", {}).get("estimated_complexity", "unknown"),
                        "quantum_transcendence_level": quantum_enhancement_result.transcendence_level,
                        "consciousness_emergence_score": quantum_enhancement_result.consciousness_emergence_score,
                        "transcendent_sql_optimization_score": transcendent_optimization.optimization_score,
                        "breakthrough_insights_count": len(quantum_enhancement_result.breakthrough_insights),
                    },
                    "validation_results": {
                        "input_validation": {
                            "score": validation_result.validation_score,
                            "issues": [{
                                "severity": issue.severity.value,
                                "type": issue.validation_type.value,
                                "message": issue.message,
                                "suggestion": issue.suggestion,
                            } for issue in validation_result.issues],
                        },
                        "sql_validation": {
                            "score": sql_validation_result.validation_score,
                            "issues": [{
                                "severity": issue.severity.value,
                                "type": issue.validation_type.value,
                                "message": issue.message,
                                "suggestion": issue.suggestion,
                            } for issue in sql_validation_result.issues],
                        },
                    },
                    "optimization_results": {
                        "optimization_score": optimization_result.get("optimization_score", 0.0),
                        "optimizations_applied": optimization_result.get("optimizations_applied", []),
                        "performance_suggestions": optimization_result.get("suggestions", []),
                        "estimated_performance_gain": optimization_result.get("estimated_performance_gain", 0.0),
                    },
                    "intelligent_optimization": {
                        "strategy": intelligent_optimization.get("execution_strategy", "direct"),
                        "complexity_score": intelligent_optimization.get("complexity_score", 0.0),
                        "optimization_score": intelligent_optimization.get("optimization_score", 0.0),
                        "performance_gain": intelligent_optimization.get("performance_gain", 0.0),
                        "optimizations_applied": intelligent_optimization.get("optimizations_applied", []),
                    },
                    "agent_output": result,
                }

        except Exception as e:
            generation_time = time.time() - start_time

            # Use transcendent error handling for failures
            transcendent_recovery = await handle_transcendent_error(
                e, 
                context={
                    "operation": "sql_generation",
                    "query": natural_language_query[:100],
                    "model": self.model_name,
                    "generation_time": generation_time
                }
            )

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
                "transcendent_recovery_applied": transcendent_recovery.get("transcendent_recovery_achieved", False),
                "recovery_score": transcendent_recovery.get("recovery_score", 0.0),
                "metadata": {
                    "model_used": self.model_name,
                    "original_query": natural_language_query,
                    "timestamp": time.time(),
                    "transcendent_error_handling": transcendent_recovery
                },
            }

    @optimize_operation("sql_execution")
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
            with error_context(
                "sql_execution",
                global_error_manager,
                {"query_length": len(sql_query), "limit": limit},
            ):
                # Advanced SQL validation before execution
                schema_info = self.get_schema_info()
                sql_validation_result = global_validator.validate_sql(sql_query, schema_info)

                if not sql_validation_result.is_valid:
                    error_issues = [issue for issue in sql_validation_result.issues
                                  if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
                    error_messages = [issue.message for issue in error_issues]
                    msg = f"SQL validation failed: {'; '.join(error_messages)}"
                    raise ValueError(msg)

                # Legacy security validation
                is_safe, violations = self.security_validator.audit_generated_query(sql_query)
                if not is_safe:
                    violation_reasons = [str(v) for v in violations]
                    msg = f"Security validation failed: {'; '.join(violation_reasons)}"
                    raise ValueError(msg)

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
                            "validation_metadata": {
                                "validation_score": sql_validation_result.validation_score,
                                "query_characteristics": sql_validation_result.metadata.get("characteristics", {}),
                                "estimated_result_size": sql_validation_result.metadata.get("estimated_rows", "unknown"),
                            },
                            "performance_metadata": {
                                "optimization_score": 0.0,
                                "optimizations_applied": [],
                                "performance_suggestions": [],
                            },
                        }
                    # Non-SELECT query (shouldn't happen with current restrictions)
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "message": "Query executed successfully (no results returned)",
                        "execution_time": execution_time,
                        "query_executed": limited_query,
                        "validation_metadata": {
                            "validation_score": sql_validation_result.validation_score,
                            "query_characteristics": sql_validation_result.metadata.get("characteristics", {}),
                        },
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
        """Get comprehensive agent performance metrics including error statistics.

        Returns:
            Dictionary containing performance metrics and error statistics
        """
        base_metrics = self.metrics.get_summary()
        error_stats = global_error_manager.get_error_statistics()
        performance_stats = global_profiler.get_performance_summary()
        resource_stats = global_profiler.get_resource_summary()
        optimization_stats = global_query_optimizer.get_cache_stats()

        # Get intelligent performance insights
        intelligent_insights = global_performance_engine.get_performance_insights()
        scaling_recommendation = global_performance_engine.get_scaling_recommendation()

        return {
            **base_metrics,
            "error_statistics": error_stats,
            "performance_statistics": performance_stats,
            "resource_utilization": resource_stats,
            "optimization_statistics": optimization_stats,
            "reliability_score": self._calculate_reliability_score(base_metrics, error_stats),
            "performance_score": self._calculate_performance_score(performance_stats, resource_stats),
            "intelligent_insights": intelligent_insights,
            "scaling_recommendation": {
                "action": scaling_recommendation.action,
                "confidence": scaling_recommendation.confidence,
                "target_instances": scaling_recommendation.target_instances,
                "reasoning": scaling_recommendation.reasoning,
                "cost_impact": scaling_recommendation.cost_impact,
                "risk_assessment": scaling_recommendation.risk_assessment,
            },
        }

    def _calculate_reliability_score(self, metrics: dict[str, Any], error_stats: dict[str, Any]) -> float:
        """Calculate overall reliability score.

        Args:
            metrics: Performance metrics
            error_stats: Error statistics

        Returns:
            Reliability score between 0.0 and 1.0
        """
        total_operations = metrics.get("total_generations", 0) + metrics.get("total_executions", 0)
        if total_operations == 0:
            return 1.0

        total_errors = error_stats.get("total_errors", 0)
        total_errors / total_operations

        # Base reliability from success rates
        gen_success_rate = metrics.get("generation_success_rate", 1.0)
        exec_success_rate = metrics.get("execution_success_rate", 1.0)
        base_reliability = (gen_success_rate + exec_success_rate) / 2

        # Penalize for recent errors
        recent_errors = error_stats.get("recent_errors_1h", 0)
        recent_penalty = min(recent_errors * 0.05, 0.3)  # Max 30% penalty

        return max(0.0, base_reliability - recent_penalty)

    def _calculate_performance_score(self, perf_stats: dict[str, Any], resource_stats: dict[str, Any]) -> float:
        """Calculate overall performance score.

        Args:
            perf_stats: Performance statistics
            resource_stats: Resource utilization statistics

        Returns:
            Performance score between 0.0 and 1.0
        """
        if perf_stats.get("message") or resource_stats.get("message"):
            return 0.5  # Default score when no data available

        # Base performance score from operation timings
        avg_duration = perf_stats.get("duration_stats", {}).get("avg", 5.0)
        duration_score = max(0.0, 1.0 - (avg_duration / 10.0))  # Penalize if > 10s avg

        # Resource efficiency score
        avg_cpu = resource_stats.get("avg_cpu_percent", 50) / 100.0
        avg_memory = resource_stats.get("avg_memory_percent", 50) / 100.0
        resource_score = max(0.0, 1.0 - max(avg_cpu, avg_memory))

        # Success rate contribution
        success_rate = perf_stats.get("success_rate", 0.9)

        # Weighted combination
        performance_score = (duration_score * 0.4 + resource_score * 0.3 + success_rate * 0.3)

        return min(1.0, performance_score)

    def _validate_input(self, query: str) -> None:
        """Validate natural language input.

        Args:
            query: Natural language query to validate

        Raises:
            ValueError: If validation fails
        """
        if not query or not query.strip():
            msg = "Query cannot be empty"
            raise ValueError(msg)

        if len(query) > self.MAX_QUERY_LENGTH:
            msg = f"Query too long (max {self.MAX_QUERY_LENGTH} characters)"
            raise ValueError(msg)

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
                return self.agent.run(query)
            except Exception as e:
                last_error = e
                logger.warning("Attempt %d failed: %s", attempt, str(e))
                if attempt < self.max_retries:
                    time.sleep(1)  # Brief delay before retry

        # Enhanced error reporting with context
        error_context_dict = {
            "operation": "sql_generation",
            "attempts": self.max_retries,
            "query_preview": query[:100],
            "model": self.model_name,
        }

        if last_error:
            global_error_manager.handle_error(
                last_error,
                "sql_generation_final_failure",
                error_context_dict,
            )

        msg = f"SQL generation failed after {self.max_retries} attempts: {last_error}"
        raise RuntimeError(msg)

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
        lines = result.split("\n")
        sql_lines = []
        in_sql = False

        for line in lines:
            line_stripped = line.strip()

            # Look for SQL keywords to identify SQL content
            if any(keyword in line_stripped.upper() for keyword in ["SELECT", "WITH", "EXPLAIN"]):
                in_sql = True
                sql_lines.append(line_stripped)
            elif in_sql and line_stripped:
                # Continue collecting SQL lines
                if line_stripped.endswith(";"):
                    sql_lines.append(line_stripped)
                    break
                sql_lines.append(line_stripped)
            elif in_sql and not line_stripped:
                # Empty line might end SQL block
                break

        if not sql_lines:
            # Fallback: look for any SQL-like content
            for line in lines:
                if "SELECT" in line.upper():
                    return line.strip()
            msg = "No SQL query found in agent response"
            raise ValueError(msg)

        sql_query = " ".join(sql_lines)

        # Clean up the SQL
        sql_query = sql_query.strip()
        if not sql_query.endswith(";"):
            sql_query += ";"

        return sql_query

    def _validate_generated_sql(self, sql_query: str) -> None:
        """Validate generated SQL query.

        Args:
            sql_query: SQL query to validate

        Raises:
            ValueError: If validation fails
        """
        if not sql_query or not sql_query.strip():
            msg = "Generated SQL is empty"
            raise ValueError(msg)

        sql_upper = sql_query.upper().strip()

        # Check if query starts with allowed operations
        allowed_start = any(sql_upper.startswith(op) for op in self.SUPPORTED_OPERATIONS)
        if not allowed_start:
            operations_str = ", ".join(self.SUPPORTED_OPERATIONS)
            error_msg = f"Query must start with one of: {operations_str}. Got: {sql_upper[:50]}..."
            global_error_manager.handle_error(
                ValueError(error_msg),
                "sql_validation",
                {"query_start": sql_upper[:50], "allowed_operations": list(self.SUPPORTED_OPERATIONS)},
            )
            raise ValueError(error_msg)

        # Check for dangerous operations
        dangerous_keywords = [
            "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER",
            "TRUNCATE", "EXEC", "EXECUTE", "CALL",
        ]
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                error_msg = f"Dangerous operation not allowed: {keyword}"
                global_error_manager.handle_error(
                    ValueError(error_msg),
                    "dangerous_sql_operation",
                    {"dangerous_keyword": keyword, "query_preview": sql_query[:100]},
                )
                raise ValueError(error_msg)

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
        if sql_query.rstrip().endswith(";"):
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
            msg = f"Agent creation failed: {e}"
            raise RuntimeError(msg) from e
