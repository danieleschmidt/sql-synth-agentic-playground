"""Streamlit UI interface for the SQL synthesis agent.

This module provides a user-friendly Streamlit interface for demonstrating
and interacting with the SQL synthesis agent capabilities.
"""

import logging
from typing import Any, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


class StreamlitUI:
    """Streamlit user interface for SQL synthesis agent."""

    def __init__(self, db_manager: Any) -> None:
        """Initialize StreamlitUI with database manager.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db_manager = db_manager
        logger.info("StreamlitUI initialized")

    def render_header(self) -> None:
        """Render the application header and description."""
        st.title("🧠 Sentiment-Aware SQL Synthesis Agent")
        st.markdown("""
        🔍 **Intelligent Natural Language to SQL Converter**

        Enter your query in plain English, and I'll analyze the emotional context 
        and intent to generate optimized SQL queries tailored to your needs.
        
        **✨ Features:**
        - 🎭 Sentiment analysis for query understanding
        - 🧠 Intent detection (analytical, exploratory, investigative)  
        - ⏰ Temporal bias detection (recent, historical, trending)
        - 📊 Magnitude bias detection (top, bottom, extreme)
        - 🛡️ Security validation and SQL injection prevention
        """)

    def render_input_form(self) -> tuple[str, bool]:
        """Render the user input form.

        Returns:
            Tuple of (user_query, submit_clicked)
        """
        st.subheader("Enter Your Query")

        user_query = st.text_area(
            "Describe what data you want to retrieve:",
            placeholder="e.g., Show me the best performing products this month, or Find problematic orders that need attention",
            height=120,
            help="Type your query in natural language. The system will analyze emotional context and intent for better SQL generation.",
        )

        submit_clicked = st.button(
            "Generate SQL",
            type="primary",
            use_container_width=True,
        )

        return user_query, submit_clicked

    def render_sql_output(self, sql_query: Optional[str]) -> None:
        """Render the generated SQL query.

        Args:
            sql_query: The generated SQL query to display
        """
        if sql_query:
            st.subheader("Generated SQL")
            st.code(sql_query, language="sql")

    def render_results(self, results: Optional[pd.DataFrame]) -> None:
        """Render query execution results.

        Args:
            results: DataFrame containing query results to display
        """
        if results is not None:
            display_query_results(results)

    def show_error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error message to display
        """
        render_error_message(message)

    def show_info(self, message: str) -> None:
        """Display an info message.

        Args:
            message: Info message to display
        """
        st.info(message)

    def show_success(self, message: str) -> None:
        """Display a success message.

        Args:
            message: Success message to display
        """
        st.success(message)
        
    def render_sentiment_analysis(self, sentiment_data: dict) -> None:
        """Render sentiment analysis results.
        
        Args:
            sentiment_data: Dictionary containing sentiment analysis results
        """
        st.subheader("🎭 Sentiment Analysis")
        
        # Create columns for layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Sentiment polarity with color coding
            polarity = sentiment_data.get("polarity", "neutral")
            polarity_colors = {
                "very_positive": "🟢",
                "positive": "🔵", 
                "neutral": "⚪",
                "negative": "🟠",
                "very_negative": "🔴"
            }
            st.metric(
                "Sentiment",
                f"{polarity_colors.get(polarity, '⚪')} {polarity.replace('_', ' ').title()}",
                f"{sentiment_data.get('compound_score', 0):.3f}"
            )
        
        with col2:
            # Intent with icon
            intent = sentiment_data.get("intent", "analytical")
            intent_icons = {
                "analytical": "📊",
                "exploratory": "🔍", 
                "investigative": "🕵️",
                "comparative": "⚖️",
                "trending": "📈",
                "problem_solving": "🛠️"
            }
            st.metric(
                "Query Intent",
                f"{intent_icons.get(intent, '📊')} {intent.replace('_', ' ').title()}",
                f"Confidence: {sentiment_data.get('confidence', 0):.2f}"
            )
            
        with col3:
            # Biases
            temporal_bias = sentiment_data.get("temporal_bias")
            magnitude_bias = sentiment_data.get("magnitude_bias") 
            
            biases = []
            if temporal_bias:
                biases.append(f"⏰ {temporal_bias}")
            if magnitude_bias:
                biases.append(f"📊 {magnitude_bias}")
            
            bias_text = " | ".join(biases) if biases else "None detected"
            st.metric("Detected Biases", bias_text)
        
        # Emotional keywords
        keywords = sentiment_data.get("emotional_keywords", [])
        if keywords:
            st.markdown(f"**🏷️ Emotional Keywords:** {', '.join(keywords)}")
        
        # Detailed scores (collapsible)
        with st.expander("📈 Detailed Sentiment Scores"):
            scores_col1, scores_col2, scores_col3 = st.columns(3)
            with scores_col1:
                st.metric("Positive", f"{sentiment_data.get('positive', 0):.3f}")
            with scores_col2:
                st.metric("Neutral", f"{sentiment_data.get('neutral', 0):.3f}")  
            with scores_col3:
                st.metric("Negative", f"{sentiment_data.get('negative', 0):.3f}")
                
        st.markdown("---")


def display_query_results(results: Optional[pd.DataFrame]) -> None:
    """Display query results in a formatted table.

    Args:
        results: DataFrame containing query results
    """
    if results is None:
        return

    st.subheader("Query Results")

    if results.empty:
        st.info("No results returned.")
        return

    # Display the dataframe
    st.dataframe(results, use_container_width=True)

    # Show summary information
    num_rows, num_cols = results.shape
    st.write(f"📊 **{num_rows}** rows x **{num_cols}** columns")


def render_error_message(message: str) -> None:
    """Render an error message with consistent formatting.

    Args:
        message: Error message to display
    """
    st.error(f"❌ Error: {message}")


def render_connection_status(db_manager: Any) -> None:
    """Render database connection status.

    Args:
        db_manager: DatabaseManager instance to test connection
    """
    st.sidebar.subheader("Database Connection")

    try:
        if db_manager.test_connection():
            st.sidebar.success("✅ Connected")
            dialect_info = db_manager.get_dialect_info()
            st.sidebar.info(f"Database: {dialect_info['name']}")
        else:
            st.sidebar.error("❌ Connection Failed")
    except Exception as e:
        st.sidebar.error(f"❌ Connection Error: {e}")


def render_sidebar_info() -> None:
    """Render informational content in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Tips")
    st.sidebar.markdown("""
    **Good query examples:**
    - "Show me all users created in 2024"
    - "Find the top 10 products by sales"
    - "List customers with no orders"

    **Be specific about:**
    - Date ranges
    - Sort order (ascending/descending)
    - Number of results (top 10, limit 50)
    - Filters and conditions
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**🔒 Security:** All queries use parameterized statements to "
        "prevent SQL injection."
    )


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="SQL Synthesis Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
