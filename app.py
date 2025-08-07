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


def simulate_sentiment_aware_sql_generation(user_query: str) -> tuple[str, dict]:
    """Simulate sentiment-aware SQL generation for demo purposes.

    Args:
        user_query: Natural language query from user

    Returns:
        Tuple of (SQL query, sentiment analysis dict)
    """
    from src.sql_synth.sentiment_analyzer import sentiment_analyzer
    
    # Perform actual sentiment analysis
    sentiment_analysis = sentiment_analyzer.analyze(user_query)
    
    # Simple demo SQL generation based on keywords and sentiment
    query_lower = user_query.lower()
    base_sql = ""

    if "users" in query_lower:
        if "active" in query_lower:
            base_sql = "SELECT * FROM users WHERE status = 'active' ORDER BY created_at DESC;"
        else:
            base_sql = "SELECT * FROM users ORDER BY created_at DESC LIMIT 100;"
    elif "orders" in query_lower:
        if "today" in query_lower or "recent" in query_lower:
            base_sql = "SELECT * FROM orders WHERE DATE(created_at) = CURRENT_DATE ORDER BY created_at DESC;"
        else:
            base_sql = "SELECT * FROM orders ORDER BY created_at DESC LIMIT 50;"
    elif "products" in query_lower:
        if "top" in query_lower or "best" in query_lower:
            base_sql = "SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id ORDER BY total_sold DESC LIMIT 10;"
        else:
            base_sql = "SELECT * FROM products WHERE active = true ORDER BY name;"
    else:
        base_sql = f"-- Query: {user_query}\nSELECT 'Sentiment-aware SQL generation' as message, 'Configure database for full functionality' as note;"
    
    # Apply sentiment-based enhancements
    enhanced_sql = sentiment_analyzer.enhance_sql_with_sentiment(base_sql, sentiment_analysis)
    
    sentiment_dict = {
        "polarity": sentiment_analysis.polarity.value,
        "confidence": sentiment_analysis.confidence,
        "compound_score": sentiment_analysis.compound_score,
        "intent": sentiment_analysis.intent.value,
        "emotional_keywords": sentiment_analysis.emotional_keywords,
        "temporal_bias": sentiment_analysis.temporal_bias,
        "magnitude_bias": sentiment_analysis.magnitude_bias,
    }
    
    return enhanced_sql, sentiment_dict


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
            with st.spinner("Analyzing sentiment and generating SQL..."):
                if demo_mode:
                    # Demo mode: simulate sentiment-aware SQL generation
                    generated_sql, sentiment_data = simulate_sentiment_aware_sql_generation(user_query)
                    ui.show_info("⚠️ Demo mode: Using actual sentiment analysis with simulated SQL generation")
                else:
                    # Real mode: would integrate with LangChain agent here
                    generated_sql, sentiment_data = simulate_sentiment_aware_sql_generation(user_query)  # Placeholder
                    ui.show_info("🧠 Sentiment-aware SQL generation active! LangChain integration in progress")

                # Display sentiment analysis
                ui.render_sentiment_analysis(sentiment_data)
                
                # Display generated SQL
                ui.render_sql_output(generated_sql)

                # Execute query and show results
                with st.spinner("Executing query..."):
                    if demo_mode:
                        # Demo mode: create fake results
                        results = create_demo_results(generated_sql)
                        ui.show_success("Demo results generated successfully!")
                    else:
                        # Real mode: would execute against database
                        results = create_demo_results(generated_sql)  # Placeholder
                        ui.show_success("Query executed successfully!")

                    # Display results
                    ui.render_results(results)

                    # Add to query history
                    st.session_state.query_history.append({
                        "user_query": user_query,
                        "generated_sql": generated_sql,
                        "timestamp": pd.Timestamp.now(),
                    })

        except Exception as e:
            logger.exception("Error processing query")
            ui.show_error(f"Failed to process query: {e}")

    # Show query history in sidebar
    if st.session_state.query_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📝 Query History")
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
            query_num = len(st.session_state.query_history) - i
            with st.sidebar.expander(f"Query {query_num}"):
                st.sidebar.write(f"**Input:** {entry['user_query'][:50]}...")
                
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
