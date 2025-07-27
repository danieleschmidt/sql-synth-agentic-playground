"""Tests for Streamlit UI interface module."""

from unittest.mock import Mock, patch

import pandas as pd

from src.sql_synth.streamlit_ui import (
    StreamlitUI,
    display_query_results,
    render_error_message,
)


class TestStreamlitUI:
    """Test StreamlitUI class."""

    @patch("src.sql_synth.streamlit_ui.st")
    def test_init(self, mock_st):
        """Test StreamlitUI initialization."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)
        assert ui.db_manager == mock_db_manager

    @patch("src.sql_synth.streamlit_ui.st")
    def test_render_header(self, mock_st):
        """Test header rendering."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)
        ui.render_header()

        mock_st.title.assert_called_once_with("SQL Synthesis Agent")
        mock_st.markdown.assert_called()

    @patch("src.sql_synth.streamlit_ui.st")
    def test_render_input_form(self, mock_st):
        """Test input form rendering."""
        mock_db_manager = Mock()
        mock_st.text_area.return_value = "Show me all users"
        mock_st.button.return_value = True

        ui = StreamlitUI(mock_db_manager)
        query, submit = ui.render_input_form()

        assert query == "Show me all users"
        assert submit is True
        mock_st.text_area.assert_called_once()
        mock_st.button.assert_called_once()

    @patch("src.sql_synth.streamlit_ui.st")
    def test_render_sql_output_with_query(self, mock_st):
        """Test SQL output rendering with query."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)

        sql_query = "SELECT * FROM users;"
        ui.render_sql_output(sql_query)

        mock_st.subheader.assert_called_with("Generated SQL")
        mock_st.code.assert_called_with(sql_query, language="sql")

    @patch("src.sql_synth.streamlit_ui.st")
    def test_render_sql_output_without_query(self, mock_st):
        """Test SQL output rendering without query."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)

        ui.render_sql_output(None)

        mock_st.subheader.assert_not_called()
        mock_st.code.assert_not_called()

    @patch("src.sql_synth.streamlit_ui.st")
    @patch("src.sql_synth.streamlit_ui.display_query_results")
    def test_render_results_with_dataframe(self, mock_display, mock_st):
        """Test results rendering with DataFrame."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)

        test_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        ui.render_results(test_df)

        mock_display.assert_called_once_with(test_df)

    @patch("src.sql_synth.streamlit_ui.st")
    def test_render_results_without_dataframe(self, mock_st):
        """Test results rendering without DataFrame."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)

        ui.render_results(None)

        # Should not call any streamlit functions for results display

    @patch("src.sql_synth.streamlit_ui.st")
    @patch("src.sql_synth.streamlit_ui.render_error_message")
    def test_show_error(self, mock_render_error, mock_st):
        """Test error display."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)

        error_msg = "Connection failed"
        ui.show_error(error_msg)

        mock_render_error.assert_called_once_with(error_msg)

    @patch("src.sql_synth.streamlit_ui.st")
    def test_show_info(self, mock_st):
        """Test info message display."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)

        info_msg = "Processing query..."
        ui.show_info(info_msg)

        mock_st.info.assert_called_once_with(info_msg)

    @patch("src.sql_synth.streamlit_ui.st")
    def test_show_success(self, mock_st):
        """Test success message display."""
        mock_db_manager = Mock()
        ui = StreamlitUI(mock_db_manager)

        success_msg = "Query executed successfully"
        ui.show_success(success_msg)

        mock_st.success.assert_called_once_with(success_msg)


class TestDisplayQueryResults:
    """Test display_query_results function."""

    @patch("src.sql_synth.streamlit_ui.st")
    def test_display_with_data(self, mock_st):
        """Test displaying DataFrame with data."""
        test_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        display_query_results(test_df)

        mock_st.subheader.assert_called_with("Query Results")
        mock_st.dataframe.assert_called_with(test_df, use_container_width=True)
        mock_st.write.assert_called()

    @patch("src.sql_synth.streamlit_ui.st")
    def test_display_empty_dataframe(self, mock_st):
        """Test displaying empty DataFrame."""
        empty_df = pd.DataFrame()
        display_query_results(empty_df)

        mock_st.subheader.assert_called_with("Query Results")
        mock_st.info.assert_called_with("No results returned.")

    @patch("src.sql_synth.streamlit_ui.st")
    def test_display_none(self, mock_st):
        """Test displaying None."""
        display_query_results(None)

        # Should not call any streamlit functions


class TestRenderErrorMessage:
    """Test render_error_message function."""

    @patch("src.sql_synth.streamlit_ui.st")
    def test_render_error(self, mock_st):
        """Test error message rendering."""
        error_msg = "Database connection failed"
        render_error_message(error_msg)

        mock_st.error.assert_called_once_with(f"❌ Error: {error_msg}")

    @patch("src.sql_synth.streamlit_ui.st")
    def test_render_empty_error(self, mock_st):
        """Test empty error message rendering."""
        render_error_message("")

        mock_st.error.assert_called_once_with("❌ Error: ")


class TestStreamlitUIIntegration:
    """Integration tests for StreamlitUI."""

    @patch("src.sql_synth.streamlit_ui.st")
    def test_complete_workflow(self, mock_st):
        """Test complete UI workflow."""
        mock_db_manager = Mock()
        mock_st.text_area.return_value = "SELECT * FROM users"
        mock_st.button.return_value = True

        ui = StreamlitUI(mock_db_manager)

        # Render header
        ui.render_header()

        # Get user input
        query, submit = ui.render_input_form()

        # Display SQL
        ui.render_sql_output("SELECT * FROM users WHERE active = 1")

        # Display results
        test_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        ui.render_results(test_df)

        # Show success
        ui.show_success("Query completed!")

        # Verify all components were called
        assert mock_st.title.called
        assert mock_st.text_area.called
        assert mock_st.button.called
        assert mock_st.code.called
        assert mock_st.success.called
