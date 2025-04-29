"""
Gradio Application Module with Agent-based Processing

This module defines the Gradio application for the text-to-SQL system
using the agent-based architecture.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from text_to_sql.agent.main import TextToSQLAgent, create_text_to_sql_agent
from text_to_sql.agent.types import AgentContext
from text_to_sql.db.base import DatabaseManager
from text_to_sql.db.postgres import PostgresDatabaseManager
from text_to_sql.llm.engine import LLMEngine
from text_to_sql.visualization.charts import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_heatmap
)
from text_to_sql.utils.config_types import AppConfig  # ...added

logger = logging.getLogger(__name__)


class AgentBasedTextToSQLApp:
    """
    Gradio application for agent-based text-to-SQL interface.
    
    This class provides a web interface for converting natural language
    to SQL using the agent-based approach, and visualizing the results.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        llm_engine: LLMEngine,
        agent_config: Dict[str, Any] = None,
        app_config: Optional[Union[Dict[str, Any], AppConfig]] = None,  # ...added
        debug_mode: bool = False
    ):
        """
        Initialize the Gradio application.
        
        Args:
            db_manager: Database manager instance
            llm_engine: LLM engine instance
            agent_config: Configuration for the agent system
            app_config: Application configuration (uses AppConfig)
            debug_mode: Whether to enable debug mode
        """
        # Setup app configuration from AppConfig
        if app_config is None:
            self.app_config = AppConfig()
        elif isinstance(app_config, dict):
            self.app_config = AppConfig.from_dict(app_config)
        else:
            self.app_config = app_config

        self.db_manager = db_manager
        self.llm_engine = llm_engine
        self.agent_config = agent_config or {}
        self.theme = self.app_config.theme or gr.themes.Base()
        self.debug_mode = debug_mode
        
        # Create the text-to-SQL agent
        self.agent = create_text_to_sql_agent(
            db_manager=db_manager,
            llm_engine=llm_engine,
            config=self.agent_config
        )
        
        # Initialize history
        self.history = []
        
        # Build the app
        self.app = None
    
    def build_app(self):
        """Build the Gradio application interface."""
        with gr.Blocks(theme=self.theme, title="Text-to-SQL Interface (Agent-Based)") as self.app:
            # Header
            gr.Markdown("# Text-to-SQL Interface (Agent-Based)")
            gr.Markdown("Ask questions about your data in natural language.")
            
            # Main components
            with gr.Row():
                # Left column - Input and schema explorer
                with gr.Column(scale=1):
                    # Query input
                    query_input = gr.Textbox(
                        label="Ask a question about your data",
                        placeholder="e.g., How many products are in each category?",
                        lines=3
                    )
                    
                    submit_btn = gr.Button("Run Query", variant="primary")
                    
                    # Database schema explorer
                    with gr.Accordion("Database Schema", open=False):
                        refresh_schema_btn = gr.Button("Refresh Schema")
                        schema_output = gr.Dataframe(
                            label="Tables and Columns",
                            headers=["Schema", "Table", "Column", "Type", "Description"],
                            wrap=True
                        )
                    
                    # History accordion
                    with gr.Accordion("Query History", open=False):
                        history_output = gr.Dataframe(
                            label="Previous Queries",
                            headers=["Timestamp", "Query", "SQL"],
                            wrap=True
                        )
                
                # Right column - Results and visualizations
                with gr.Column(scale=2):
                    # Tabs for different views
                    with gr.Tabs():
                        with gr.TabItem("Results"):
                            # Generated SQL
                            sql_output = gr.Code(
                                label="Generated SQL",
                                language="sql",
                                interactive=True
                            )
                            
                            # Execute button
                            with gr.Row():
                                execute_btn = gr.Button("Execute SQL", variant="secondary")
                                clear_btn = gr.Button("Clear", variant="secondary")
                            
                            # Results table
                            results_output = gr.Dataframe(label="Query Results")
                            
                            # Explanation
                            explanation_output = gr.Markdown(label="Explanation")
                            
                            # Status message
                            status_output = gr.Markdown()
                        
                        with gr.TabItem("Visualization"):
                            # Visualization settings
                            with gr.Row():
                                with gr.Column(scale=1):
                                    viz_type = gr.Dropdown(
                                        label="Visualization Type",
                                        choices=["Table", "Bar Chart", "Line Chart", "Scatter Plot", "Heatmap"],
                                        value="Table"
                                    )
                                    
                                with gr.Column(scale=1):
                                    x_column = gr.Dropdown(label="X-Axis Column", allow_custom_value=True)
                                    y_column = gr.Dropdown(label="Y-Axis Column", allow_custom_value=True)
                                    
                                with gr.Column(scale=1):
                                    color_column = gr.Dropdown(label="Color Column (optional)", allow_custom_value=True)
                                    agg_function = gr.Dropdown(
                                        label="Aggregation Function",
                                        choices=["None", "Count", "Sum", "Average", "Min", "Max"],
                                        value="None"
                                    )
                            
                            # Update visualization button
                            update_viz_btn = gr.Button("Update Visualization")
                            
                            # Visualization output
                            viz_output = gr.Plot(label="Visualization")
                        
                        if self.debug_mode:
                            with gr.TabItem("Agent Debug"):
                                # Agent reasoning
                                reasoning_output = gr.Markdown(label="Agent Reasoning Steps")
                                
                                # Timing information
                                timing_output = gr.JSON(label="Execution Times")
                                
                                # Raw agent context
                                with gr.Accordion("Raw Agent Context", open=False):
                                    context_output = gr.JSON(label="Agent Context")
            
            # Define event handlers
            # Build outputs list conditionally
            outputs = [
                sql_output, results_output, explanation_output, 
                status_output, x_column, y_column, color_column, 
                viz_type
            ]
            if self.debug_mode:
                outputs.extend([reasoning_output, timing_output, context_output])
            outputs.append(viz_output)

            submit_btn.click(
                fn=self.handle_query,
                inputs=[query_input],
                outputs=outputs
            )
            
            execute_btn.click(
                fn=self.execute_sql,
                inputs=[sql_output],
                outputs=[
                    results_output, status_output, 
                    x_column, y_column, color_column
                ]
            )
            
            clear_btn.click(
                fn=self.clear_results,
                inputs=[],
                outputs=[
                    results_output, sql_output, explanation_output,
                    status_output, viz_output
                ]
            )
            
            refresh_schema_btn.click(
                fn=self.refresh_schema,
                inputs=[],
                outputs=[schema_output]
            )
            
            update_viz_btn.click(
                fn=self.update_visualization,
                inputs=[results_output, viz_type, x_column, y_column, color_column, agg_function],
                outputs=[viz_output]
            )
            
            # Load schema on startup
            self.refresh_schema()
        
        return self.app
    
    def launch(self, **kwargs):
        """
        Launch the Gradio application.
        
        Args:
            **kwargs: Keyword arguments to pass to gr.launch()
        """
        if self.app is None:
            self.build_app()
        
        self.app.launch(**kwargs)
    
    def handle_query(
        self, 
        query: str
    ) -> List[Union[str, pd.DataFrame, plt.Figure, Dict[str, Any], None]]:
        """
        Handle a natural language query using the agent-based approach.
        
        Args:
            query: Natural language query
            
        Returns:
            List containing:
            - SQL query string
            - DataFrame with results
            - Explanation string
            - Status message
            - X-axis column options
            - Y-axis column options
            - Color column options
            - Visualization type
            - Reasoning steps (if debug mode)
            - Timing information (if debug mode)
            - Raw agent context (if debug mode)
            - Visualization figure
        """
        if not query.strip():
            return [
                "", None, "", "Please enter a query.", 
                gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), "Table", None
            ]
        
        try:
            # Process the query with the agent
            context, info = self.agent.process_query(query)
            
            # Extract results
            sql_query = context.sql_query
            results = context.query_results
            error = context.result_error
            
            # Convert results to DataFrame
            df = pd.DataFrame(results) if results else pd.DataFrame()
            
            # Get explanation
            explanation = context.explanations.get("results", "")
            
            # Format the status message
            if error:
                status = f"❌ Error: {error}"
            else:
                rows = len(df)
                proc_time = info["processing_time"]
                status = f"✅ Query processed in {proc_time:.2f} seconds, returned {rows} rows."
            
            # Get column names for visualization dropdowns
            columns = df.columns.tolist() if not df.empty else []
            
            # Get visualization suggestion
            viz_suggestion = context.metadata.get("visualization", {})
            viz_type_suggestion = viz_suggestion.get("type", "Table")
            
            # Format reasoning steps for debug mode (omitted from outputs)
            reasoning = ""
            if self.debug_mode:
                reasoning = "## Agent Reasoning Steps\n\n"
                reasoning += "\n".join([f"- {step}" for step in context.reasoning_steps])
            
            # Add to history
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.history.append({
                "timestamp": timestamp,
                "query": query,
                "sql": sql_query,
                "results": results,
                "explanation": explanation
            })
            
            # Create visualization if suggested
            viz_figure = None
            if not df.empty and viz_type_suggestion != "Table":
                x_col = viz_suggestion.get("x_column")
                y_col = viz_suggestion.get("y_column")
                color_col = viz_suggestion.get("color_column")
                
                if x_col in columns and y_col in columns:
                    if viz_type_suggestion == "Bar Chart":
                        viz_figure = create_bar_chart(df, x_col, y_col, color_col if color_col in columns else None)
                    elif viz_type_suggestion == "Line Chart":
                        viz_figure = create_line_chart(df, x_col, y_col, color_col if color_col in columns else None)
                    elif viz_type_suggestion == "Scatter Plot":
                        viz_figure = create_scatter_plot(df, x_col, y_col, color_col if color_col in columns else None)
                    elif viz_type_suggestion == "Heatmap":
                        viz_figure = create_heatmap(df, x_col, y_col)
            
            # Return outputs:
            # 0: Generated SQL query string
            # 1: DataFrame with results
            # 2: Explanation text
            # 3: Status message
            # 4: x_axis dropdown update
            # 5: y_axis dropdown update
            # 6: color dropdown update
            # 7: Visualization type string
            # 8: Visualization figure
            return (
                sql_query,
                df,
                explanation,
                status,
                gr.update(choices=columns),
                gr.update(choices=columns),
                gr.update(choices=columns),
                viz_type_suggestion,
                viz_figure
            )
            
        except Exception as e:
            logger.error(f"Error handling query: {e}", exc_info=True)
            return [
                "", 
                None, 
                "", 
                f"❌ Error: {str(e)}", 
                gr.update(choices=[]),
                gr.update(choices=[]),
                gr.update(choices=[]),
                "Table",
                None
            ]
    
    def execute_sql(self, sql_query: str) -> Tuple[pd.DataFrame, str, List[str], List[str], List[str]]:
        """
        Execute an SQL query.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Tuple containing:
            - DataFrame with query results
            - Status message
            - List of column names for x-axis dropdown
            - List of column names for y-axis dropdown
            - List of column names for color dropdown
        """
        if not sql_query.strip():
            return None, "Please enter an SQL query.", [], [], []
        
        try:
            start_time = time.time()
            
            # Execute the query
            results, error = self.db_manager.execute_query(sql_query)
            
            if error:
                return None, f"❌ Error executing query: {error}", [], [], []
            
            # Convert to DataFrame
            if not results:
                return pd.DataFrame(), "ℹ️ Query executed successfully but returned no results.", [], [], []
            
            df = pd.DataFrame(results)
            
            # Format the status message
            execution_time = time.time() - start_time
            row_count = len(df)
            status = f"✅ Query executed in {execution_time:.2f} seconds, returned {row_count} rows."
            
            # Get column names for visualization dropdowns
            columns = df.columns.tolist()
            
            return df, status, columns, columns, ["None"] + columns
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None, f"❌ Error: {str(e)}", [], [], []
    
    def clear_results(self) -> Tuple[None, str, str, str, None]:
        """
        Clear the query results.
        
        Returns:
            Tuple containing empty outputs
        """
        return None, "", "", "Results cleared.", None
    
    def refresh_schema(self) -> pd.DataFrame:
        """
        Refresh the database schema display.
        
        Returns:
            DataFrame with schema information
        """
        try:
            # Get the schema
            schema = self.db_manager.get_schema(refresh=True)
            
            # Convert to DataFrame format
            schema_rows = []
            
            for schema_name, tables in schema.items():
                for table_name, table_info in tables.items():
                    for column_name, column_info in table_info['columns'].items():
                        row = {
                            "Schema": schema_name,
                            "Table": table_name,
                            "Column": column_name,
                            "Type": column_info['data_type'],
                            "Description": column_info['description'] or ""
                        }
                        schema_rows.append(row)
            
            return pd.DataFrame(schema_rows)
            
        except Exception as e:
            logger.error(f"Error refreshing schema: {e}")
            return pd.DataFrame([{"Error": str(e)}])
    
    def update_visualization(
        self,
        data: Optional[pd.DataFrame],
        viz_type: str,
        x_column: Optional[str],
        y_column: Optional[str],
        color_column: Optional[str],
        agg_function: str
    ) -> Optional[plt.Figure]:
        """
        Update the visualization based on user selections.
        
        Args:
            data: DataFrame with query results
            viz_type: Type of visualization
            x_column: Column to use for x-axis
            y_column: Column to use for y-axis
            color_column: Column to use for color
            agg_function: Aggregation function to apply
            
        Returns:
            Matplotlib figure with the visualization
        """
        if data is None or data.empty:
            return None
        
        if viz_type == "Table":
            return None
        
        if not x_column or x_column not in data.columns:
            return None
        
        if not y_column or y_column not in data.columns:
            return None
        
        # Ensure color_column is a string and not a list
        if isinstance(color_column, list):
            # Filter out invalid values and ensure columns exist in the DataFrame
            filtered = [c for c in color_column if isinstance(c, str) and c != "None" and c in data.columns]
            if not filtered:
                logger.warning("No valid columns found in color_column list. Setting color_column to None.")
                color_column = None
            else:
                color_column = filtered[0]
        # Prepare color column
        if color_column == "None" or color_column not in data.columns:
            color_column = None
        
        # Apply aggregation if needed
        if agg_function != "None":
            # Group by x and color columns
            group_cols = [x_column]
            if color_column:
                group_cols.append(color_column)
            
            # Apply aggregation
            if agg_function == "Count":
                data = data.groupby(group_cols).size().reset_index(name=y_column)
            elif agg_function == "Sum":
                data = data.groupby(group_cols)[y_column].sum().reset_index()
            elif agg_function == "Average":
                data = data.groupby(group_cols)[y_column].mean().reset_index()
            elif agg_function == "Min":
                data = data.groupby(group_cols)[y_column].min().reset_index()
            elif agg_function == "Max":
                data = data.groupby(group_cols)[y_column].max().reset_index()
        
        # Create the visualization
        if viz_type == "Bar Chart":
            return create_bar_chart(data, x_column, y_column, color_column)
        elif viz_type == "Line Chart":
            return create_line_chart(data, x_column, y_column, color_column)
        elif viz_type == "Scatter Plot":
            return create_scatter_plot(data, x_column, y_column, color_column)
        elif viz_type == "Heatmap":
            return create_heatmap(data, x_column, y_column)
        
        return None