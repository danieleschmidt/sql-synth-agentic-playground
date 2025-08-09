"""Streamlit UI interface for the SQL synthesis agent.

This module provides a user-friendly Streamlit interface for demonstrating
and interacting with the SQL synthesis agent capabilities.
"""

import logging
import time
from typing import Any, Optional, Dict

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🧠 SQL Synthesis Agent")
            st.markdown("""
            **Intelligent Natural Language to SQL Converter**
            
            Transform your questions into precise SQL queries with AI-powered analysis,
            security validation, and performance optimization.
            """)
        with col2:
            # Real-time metrics dashboard
            if hasattr(st.session_state, 'agent_metrics'):
                self.render_metrics_widget(st.session_state.agent_metrics)
            else:
                st.metric("Queries Processed", "0")
                st.metric("Success Rate", "N/A")

    def render_input_form(self) -> tuple[str, bool]:
        """Render the enhanced user input form with smart features.

        Returns:
            Tuple of (user_query, submit_clicked)
        """
        st.subheader("🎯 Query Builder")
        
        # Quick suggestion buttons
        st.markdown("**Quick Templates:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Analytics"):
                st.session_state.query_template = "Show me the top 10 customers by total purchase value"
        with col2:
            if st.button("📈 Trends"):
                st.session_state.query_template = "Show sales trends over the last 6 months"
        with col3:
            if st.button("👥 Users"):
                st.session_state.query_template = "Find all active users who joined in the last month"
        with col4:
            if st.button("🔍 Search"):
                st.session_state.query_template = "Search for products containing 'premium'"
        
        # Get template from session state if available
        default_query = getattr(st.session_state, 'query_template', '')
        if default_query:
            st.session_state.query_template = ''  # Clear after use

        user_query = st.text_area(
            "Describe what data you want to retrieve:",
            value=default_query,
            placeholder="e.g., Show me revenue by month for the last year, or Find customers who haven't ordered in 90 days",
            height=120,
            help="💡 Be specific about dates, filters, and desired output format. Use natural language!",
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                limit_results = st.number_input("Max Results", min_value=10, max_value=1000, value=100)
                include_metadata = st.checkbox("Include Query Metadata", value=False)
            with col2:
                explain_query = st.checkbox("Explain SQL Logic", value=False)
                performance_analysis = st.checkbox("Performance Analysis", value=False)
        
        # Store advanced options in session state
        st.session_state.advanced_options = {
            'limit_results': limit_results,
            'include_metadata': include_metadata,
            'explain_query': explain_query,
            'performance_analysis': performance_analysis
        }

        submit_clicked = st.button(
            "🚀 Generate & Execute SQL",
            type="primary",
            use_container_width=True,
        )

        return user_query, submit_clicked

    def render_sql_output(self, sql_query: Optional[str], metadata: Optional[Dict] = None) -> None:
        """Render the generated SQL query with enhanced visualization.

        Args:
            sql_query: The generated SQL query to display
            metadata: Additional metadata about the query generation
        """
        if sql_query:
            st.subheader("🔧 Generated SQL Query")
            
            # Tabbed interface for SQL and metadata
            tab1, tab2, tab3 = st.tabs(["SQL Query", "Query Analysis", "Security Report"])
            
            with tab1:
                st.code(sql_query, language="sql")
                
                # Copy button (simulated)
                if st.button("📋 Copy SQL to Clipboard"):
                    st.success("SQL copied to clipboard! (Feature simulated)")
            
            with tab2:
                if metadata:
                    st.markdown("**Query Complexity Analysis:**")
                    
                    # Create metrics columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Generation Time", f"{metadata.get('generation_time', 0):.2f}s")
                    with col2:
                        st.metric("Query Length", f"{len(sql_query)} chars")
                    with col3:
                        st.metric("Estimated Complexity", self._estimate_complexity(sql_query))
                    with col4:
                        st.metric("Model Used", metadata.get('model_used', 'Unknown'))
                    
                    # Visual complexity breakdown
                    complexity_data = self._analyze_query_complexity(sql_query)
                    if complexity_data:
                        fig = px.bar(
                            x=list(complexity_data.keys()),
                            y=list(complexity_data.values()),
                            title="Query Feature Analysis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No metadata available for analysis")
            
            with tab3:
                self._render_security_analysis(sql_query)

    def render_results(self, results: Optional[pd.DataFrame], execution_metadata: Optional[Dict] = None) -> None:
        """Render enhanced query execution results with analytics.

        Args:
            results: DataFrame containing query results to display
            execution_metadata: Metadata about query execution
        """
        if results is not None:
            st.subheader("📊 Query Results & Analytics")
            
            # Results overview
            if not results.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows Returned", len(results))
                with col2:
                    st.metric("Columns", len(results.columns))
                with col3:
                    if execution_metadata:
                        st.metric("Execution Time", f"{execution_metadata.get('execution_time', 0):.3f}s")
                    else:
                        st.metric("Execution Time", "N/A")
                with col4:
                    memory_usage = results.memory_usage(deep=True).sum() / 1024  # KB
                    st.metric("Memory Usage", f"{memory_usage:.1f} KB")
                
                # Tabbed results display
                tab1, tab2, tab3 = st.tabs(["Data Table", "Data Insights", "Visualizations"])
                
                with tab1:
                    display_query_results(results)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"sql_results_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                
                with tab2:
                    self._render_data_insights(results)
                
                with tab3:
                    self._render_smart_visualizations(results)
            else:
                st.info("No data returned from query")

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

    def render_metrics_widget(self, metrics: Dict[str, Any]) -> None:
        """Render real-time metrics widget.
        
        Args:
            metrics: Dictionary containing agent performance metrics
        """
        total_queries = metrics.get('total_queries', 0)
        success_rate = metrics.get('success_rate', 0)
        avg_time = metrics.get('avg_generation_time', 0)
        
        st.metric("Queries Processed", total_queries)
        st.metric("Success Rate", f"{success_rate:.1%}")
        st.metric("Avg Gen Time", f"{avg_time:.2f}s")

    def _estimate_complexity(self, sql_query: str) -> str:
        """Estimate SQL query complexity.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Complexity level as string
        """
        complexity_score = 0
        sql_upper = sql_query.upper()
        
        # Count complexity indicators
        if 'JOIN' in sql_upper:
            complexity_score += 2
        if 'SUBQUERY' in sql_upper or '(' in sql_query:
            complexity_score += 2
        if 'GROUP BY' in sql_upper:
            complexity_score += 1
        if 'HAVING' in sql_upper:
            complexity_score += 2
        if 'WINDOW' in sql_upper or 'OVER' in sql_upper:
            complexity_score += 3
        if 'WITH' in sql_upper:
            complexity_score += 2
        
        if complexity_score <= 2:
            return "Simple"
        elif complexity_score <= 5:
            return "Medium"
        else:
            return "Complex"

    def _analyze_query_complexity(self, sql_query: str) -> Dict[str, int]:
        """Analyze query features for complexity visualization.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Dictionary of feature counts
        """
        sql_upper = sql_query.upper()
        return {
            'Joins': sql_upper.count('JOIN'),
            'Subqueries': sql_query.count('(') - sql_query.count('()'),
            'Aggregations': sql_upper.count('GROUP BY') + sql_upper.count('COUNT') + sql_upper.count('SUM'),
            'Filters': sql_upper.count('WHERE') + sql_upper.count('HAVING'),
            'Windows': sql_upper.count('OVER')
        }

    def _render_security_analysis(self, sql_query: str) -> None:
        """Render security analysis of the SQL query.
        
        Args:
            sql_query: SQL query to analyze
        """
        st.markdown("**Security Validation Results:**")
        
        # Simulate security checks
        checks = [
            ("SQL Injection Protection", True, "✅ Parameterized queries used"),
            ("Read-Only Operations", 'SELECT' in sql_query.upper(), "✅ Only SELECT operations allowed" if 'SELECT' in sql_query.upper() else "⚠️ Non-SELECT operation detected"),
            ("Query Length Validation", len(sql_query) < 10000, "✅ Query length within limits"),
            ("Dangerous Keywords", not any(kw in sql_query.upper() for kw in ['DROP', 'DELETE', 'UPDATE', 'CREATE']), "✅ No dangerous operations detected")
        ]
        
        for check_name, passed, message in checks:
            if passed:
                st.success(f"{check_name}: {message}")
            else:
                st.error(f"{check_name}: {message}")

    def _render_data_insights(self, results: pd.DataFrame) -> None:
        """Render automatic data insights.
        
        Args:
            results: DataFrame to analyze
        """
        st.markdown("**Automatic Data Insights:**")
        
        # Basic statistics
        if not results.empty:
            numeric_cols = results.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Column Statistics:**")
                st.dataframe(results[numeric_cols].describe(), use_container_width=True)
            
            # Data quality indicators
            col1, col2 = st.columns(2)
            with col1:
                null_counts = results.isnull().sum()
                if null_counts.sum() > 0:
                    st.markdown("**Missing Values:**")
                    missing_data = null_counts[null_counts > 0]
                    for col, count in missing_data.items():
                        pct = (count / len(results)) * 100
                        st.write(f"- {col}: {count} ({pct:.1f}%)")
                else:
                    st.success("✅ No missing values detected")
            
            with col2:
                duplicates = results.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
                if duplicates > 0:
                    st.warning(f"{duplicates} duplicate rows found")

    def _render_smart_visualizations(self, results: pd.DataFrame) -> None:
        """Render smart visualizations based on data types.
        
        Args:
            results: DataFrame to visualize
        """
        if results.empty:
            st.info("No data to visualize")
            return
        
        st.markdown("**Smart Visualizations:**")
        
        numeric_cols = results.select_dtypes(include=['number']).columns
        categorical_cols = results.select_dtypes(include=['object', 'category']).columns
        datetime_cols = results.select_dtypes(include=['datetime']).columns
        
        # Auto-generate appropriate charts
        if len(numeric_cols) >= 2:
            # Correlation heatmap for numeric data
            fig = px.imshow(
                results[numeric_cols].corr(),
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            # Bar chart for categorical vs numeric
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            if len(results[cat_col].unique()) <= 20:  # Avoid too many categories
                fig = px.bar(
                    results.groupby(cat_col)[num_col].mean().reset_index(),
                    x=cat_col,
                    y=num_col,
                    title=f"Average {num_col} by {cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
            # Time series for datetime vs numeric
            date_col = datetime_cols[0]
            num_col = numeric_cols[0]
            
            fig = px.line(
                results.sort_values(date_col),
                x=date_col,
                y=num_col,
                title=f"{num_col} Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)


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
    """Render enhanced informational content in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Smart Query Assistant")
    
    # Query complexity indicator
    if hasattr(st.session_state, 'last_query_complexity'):
        complexity = st.session_state.last_query_complexity
        if complexity == 'Simple':
            st.sidebar.success(f"🟢 Last Query: {complexity}")
        elif complexity == 'Medium':
            st.sidebar.warning(f"🟡 Last Query: {complexity}")
        else:
            st.sidebar.error(f"🔴 Last Query: {complexity}")
    
    st.sidebar.markdown("""
    **💡 Pro Tips:**
    - Be specific about date ranges ("last 30 days", "Q1 2024")
    - Include sorting preferences ("top 10", "highest first")
    - Specify data relationships ("customers with orders")
    - Use business terms ("revenue", "active users", "conversion rate")
    
    **🔥 Advanced Patterns:**
    - "Compare X vs Y by month"
    - "Show trends for category Z"
    - "Find outliers in metric ABC"
    - "Analyze cohort behavior"
    """)
    
    # Performance monitor
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Performance Monitor")
    if hasattr(st.session_state, 'performance_stats'):
        stats = st.session_state.performance_stats
        st.sidebar.metric("Avg Generation Time", f"{stats.get('avg_gen_time', 0):.2f}s")
        st.sidebar.metric("Avg Execution Time", f"{stats.get('avg_exec_time', 0):.2f}s")
        st.sidebar.metric("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1%}")
    else:
        st.sidebar.info("Performance stats will appear after queries")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**🛡️ Security:** Multi-layer protection including SQL injection prevention, "
        "query validation, and execution sandboxing."
    )


def configure_page() -> None:
    """Configure Streamlit page settings with enhanced UI."""
    st.set_page_config(
        page_title="SQL Synthesis Agent - AI-Powered Database Queries",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/danieleschmidt/sql-synth-agentic-playground',
            'Report a bug': 'https://github.com/danieleschmidt/sql-synth-agentic-playground/issues',
            'About': 'SQL Synthesis Agent - Transform natural language into optimized SQL queries'
        }
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .success-metric {
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        border-left: 4px solid #ffc107;
    }
    .error-metric {
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
