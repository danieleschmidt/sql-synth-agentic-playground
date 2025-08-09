"""Main Streamlit application for SQL Synthesis Agent.

This is the entry point for the Streamlit web application that provides
an interactive interface for converting natural language to SQL.
"""

import logging
from typing import Optional

import pandas as pd
import streamlit as st

from src.sql_synth.database import DatabaseManager, get_database_manager
from src.sql_synth.streamlit_ui import (
    StreamlitUI,
    configure_page,
    render_connection_status,
    render_sidebar_info,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = None
    if "ui" not in st.session_state:
        st.session_state.ui = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def setup_database_connection() -> Optional[DatabaseManager]:
    """Set up database connection from environment variables.

    Returns:
        DatabaseManager instance if successful, None otherwise
    """
    try:
        db_manager = get_database_manager()
    except Exception as e:
        logger.exception("Failed to create database manager")
        st.error(f"Database configuration error: {e}")
        st.info("Please check your environment variables in .env file")
        return None
    else:
        logger.info("Database manager created successfully")
        return db_manager


def demo_mode_warning() -> None:
    """Show demo mode warning if no database is configured."""
    st.warning("""
    🚧 **Demo Mode**: No database connection configured.

    To use the full functionality:
    1. Create a `.env` file based on `.env.example`
    2. Set your database connection details
    3. Restart the application
    """)


def simulate_sql_generation(user_query: str) -> str:
    """Simulate SQL generation for demo purposes.

    Args:
        user_query: Natural language query from user

    Returns:
        Simulated SQL query
    """
    # Simple demo SQL generation based on keywords
    query_lower = user_query.lower()

    if "users" in query_lower:
        if "active" in query_lower:
            return (
                "SELECT * FROM users WHERE status = 'active' "
                "ORDER BY created_at DESC;"
            )
        return "SELECT * FROM users ORDER BY created_at DESC LIMIT 100;"
    if "orders" in query_lower:
        if "today" in query_lower or "recent" in query_lower:
            return (
                "SELECT * FROM orders WHERE DATE(created_at) = CURRENT_DATE "
                "ORDER BY created_at DESC;"
            )
        return "SELECT * FROM orders ORDER BY created_at DESC LIMIT 50;"
    if "products" in query_lower:
        if "top" in query_lower or "best" in query_lower:
            return (
                "SELECT p.name, SUM(oi.quantity) as total_sold FROM products p "
                "JOIN order_items oi ON p.id = oi.product_id "
                "GROUP BY p.id ORDER BY total_sold DESC LIMIT 10;"
            )
        return "SELECT * FROM products WHERE active = true ORDER BY name;"
    return (
        f"-- Query: {user_query}\n"
        "SELECT 'Demo query generation' as message, "
        "'Configure database connection for real SQL synthesis' as note;"
    )


def create_demo_results(sql_query: str) -> pd.DataFrame:
    """Create demo results for display.

    Args:
        sql_query: SQL query to create demo results for

    Returns:
        Demo DataFrame with sample data
    """
    if "users" in sql_query.lower():
        return pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice Johnson", "Bob Smith", "Carol Davis"],
            "email": ["alice@example.com", "bob@example.com", "carol@example.com"],
            "status": ["active", "active", "inactive"],
            "created_at": ["2024-01-15", "2024-01-20", "2024-01-10"],
        })
    if "orders" in sql_query.lower():
        return pd.DataFrame({
            "id": [101, 102, 103],
            "user_id": [1, 2, 1],
            "total": [25.99, 45.50, 15.00],
            "status": ["completed", "pending", "completed"],
            "created_at": ["2024-01-25", "2024-01-25", "2024-01-24"],
        })
    if "products" in sql_query.lower():
        return pd.DataFrame({
            "name": ["Laptop", "Mouse", "Keyboard"],
            "total_sold": [15, 45, 30],
            "category": ["Electronics", "Accessories", "Accessories"],
        })
    return pd.DataFrame({
        "message": ["Demo query generation"],
        "note": ["Configure database connection for real SQL synthesis"],
    })


def main() -> None:
    """Main application function."""
    # Configure page
    configure_page()

    # Initialize session state
    initialize_session_state()

    # Try to set up database connection
    if st.session_state.db_manager is None:
        st.session_state.db_manager = setup_database_connection()

    # Initialize UI
    if st.session_state.ui is None and st.session_state.db_manager is not None:
        st.session_state.ui = StreamlitUI(st.session_state.db_manager)

    # Check if we're in demo mode
    demo_mode = st.session_state.db_manager is None

    # Render sidebar
    render_sidebar_info()

    if not demo_mode:
        render_connection_status(st.session_state.db_manager)
    else:
        st.sidebar.error("❌ No Database Connection")
        st.sidebar.markdown("**Demo Mode Active**")

    # Main content
    if demo_mode:
        demo_mode_warning()
        # Create a demo UI instance
        ui = StreamlitUI(None)
    else:
        ui = st.session_state.ui

    # Render header
    ui.render_header()

    # Render input form
    user_query, submit_clicked = ui.render_input_form()

    # Process query when submitted
    if submit_clicked and user_query.strip():
        try:
            with st.spinner("Generating SQL..."):
                if demo_mode:
                    # Demo mode: simulate SQL generation
                    generated_sql = simulate_sql_generation(user_query)
                    ui.show_info("⚠️ Demo mode: SQL generated using simple rules")
                else:
                    # Real mode: Use actual SQL synthesis agent
                    from src.sql_synth.agent import AgentFactory
                    try:
                        agent = AgentFactory.create_agent(st.session_state.db_manager)
                        result = agent.generate_sql(user_query)
                        if result["success"]:
                            generated_sql = result["sql_query"]
                            ui.show_success(f"SQL generated in {result['generation_time']:.2f}s")
                        else:
                            ui.show_error(f"SQL generation failed: {result['error']}")
                            generated_sql = None
                    except Exception as e:
                        logger.exception("Agent creation failed")
                        ui.show_error(f"Agent initialization failed: {e}")
                        generated_sql = simulate_sql_generation(user_query)

                # Display generated SQL with metadata
                metadata = result.get('metadata', {}) if not demo_mode and isinstance(result, dict) else {}
                ui.render_sql_output(generated_sql, metadata)

                # Execute query and show results
                if generated_sql:
                    with st.spinner("Executing query..."):
                        if demo_mode:
                            # Demo mode: create fake results
                            results = create_demo_results(generated_sql)
                            ui.show_success("Demo results generated successfully!")
                        else:
                            # Real mode: execute against database using agent
                            try:
                                from src.sql_synth.agent import AgentFactory
                                agent = AgentFactory.create_agent(st.session_state.db_manager)
                                exec_result = agent.execute_sql(generated_sql)
                                if exec_result["success"]:
                                    if "rows" in exec_result:
                                        results = pd.DataFrame(exec_result["rows"])
                                        ui.show_success(f"Query executed successfully! ({exec_result['row_count']} rows in {exec_result['execution_time']:.2f}s)")
                                    else:
                                        results = pd.DataFrame({"message": [exec_result.get("message", "Query executed successfully")]})
                                        ui.show_success("Query executed successfully!")
                                else:
                                    ui.show_error(f"Query execution failed: {exec_result['error']}")
                                    results = None
                            except Exception as e:
                                logger.exception("Query execution failed")
                                ui.show_error(f"Execution failed: {e}")
                                results = None
                else:
                    results = None

                    # Display results with execution metadata
                    execution_metadata = exec_result if not demo_mode and 'exec_result' in locals() else {}
                    ui.render_results(results, execution_metadata)
                    
                    # Update session state with metrics and performance data
                    if not demo_mode and 'agent' in locals():
                        st.session_state.agent_metrics = agent.get_metrics()
                        st.session_state.last_query_complexity = ui._estimate_complexity(generated_sql)
                        
                        # Update performance stats
                        if not hasattr(st.session_state, 'performance_stats'):
                            st.session_state.performance_stats = {
                                'total_queries': 0,
                                'total_gen_time': 0,
                                'total_exec_time': 0,
                                'cache_hits': 0
                            }
                        
                        stats = st.session_state.performance_stats
                        stats['total_queries'] += 1
                        if 'result' in locals() and isinstance(result, dict):
                            stats['total_gen_time'] += result.get('generation_time', 0)
                        if 'exec_result' in locals() and isinstance(exec_result, dict):
                            stats['total_exec_time'] += exec_result.get('execution_time', 0)
                        
                        # Calculate averages
                        stats['avg_gen_time'] = stats['total_gen_time'] / stats['total_queries']
                        stats['avg_exec_time'] = stats['total_exec_time'] / stats['total_queries']
                        stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']

                    # Add to enhanced query history
                    history_entry = {
                        "user_query": user_query,
                        "generated_sql": generated_sql,
                        "timestamp": pd.Timestamp.now(),
                        "success": True,
                        "row_count": len(results) if results is not None else 0,
                        "complexity": getattr(st.session_state, 'last_query_complexity', 'Unknown'),
                        "generation_time": result.get('generation_time', 0) if not demo_mode and isinstance(result, dict) else 0,
                        "execution_time": exec_result.get('execution_time', 0) if not demo_mode and 'exec_result' in locals() and isinstance(exec_result, dict) else 0
                    }
                    st.session_state.query_history.append(history_entry)

        except Exception as e:
            logger.exception("Error processing query")
            ui.show_error(f"Failed to process query: {e}")

    # Show enhanced query history in sidebar
    if st.session_state.query_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📝 Recent Queries")
        
        # Summary metrics
        total_queries = len(st.session_state.query_history)
        successful_queries = sum(1 for entry in st.session_state.query_history if entry.get('success', True))
        avg_gen_time = sum(entry.get('generation_time', 0) for entry in st.session_state.query_history) / total_queries
        
        st.sidebar.metric("Total Queries", total_queries)
        st.sidebar.metric("Success Rate", f"{successful_queries/total_queries:.1%}")
        st.sidebar.metric("Avg Gen Time", f"{avg_gen_time:.2f}s")
        
        # Recent queries
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
            query_num = len(st.session_state.query_history) - i
            complexity = entry.get('complexity', 'Unknown')
            success_icon = "✅" if entry.get('success', True) else "❌"
            
            with st.sidebar.expander(f"{success_icon} Query {query_num} ({complexity})"):
                st.sidebar.write(f"**Input:** {entry['user_query'][:50]}...")
                st.sidebar.write(f"**Rows:** {entry.get('row_count', 0)}")
                st.sidebar.write(f"**Time:** {entry.get('generation_time', 0):.2f}s gen + {entry.get('execution_time', 0):.2f}s exec")
                
                # Truncate long SQL queries
                MAX_SQL_DISPLAY = 100
                sql_display = (
                    entry["generated_sql"][:MAX_SQL_DISPLAY] + "..."
                    if len(entry["generated_sql"]) > MAX_SQL_DISPLAY
                    else entry["generated_sql"]
                )
                st.sidebar.code(sql_display, language="sql")


if __name__ == "__main__":
    main()
