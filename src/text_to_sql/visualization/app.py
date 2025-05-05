"""
Gradio Application Module

This module defines the Gradio application for the text-to-SQL system.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from text_to_sql.db.base import DatabaseManager
from text_to_sql.llm.engine import LLMEngine
from text_to_sql.llm.semantic import SemanticEngine
from text_to_sql.visualization.charts import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_heatmap
)
from text_to_sql.utils.config_types import AppConfig  # ...added

logger = logging.getLogger(__name__)


class TextToSQLApp:
    """
    Gradio application for text-to-SQL interface.
    
    This class provides a web interface for converting natural language
    to SQL and visualizing the results.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        llm_engine: LLMEngine,
        semantic_engine: Optional[SemanticEngine] = None,
        app_config: Optional[Union[Dict[str, Any], AppConfig]] = None,  # ...added
        debug_mode: bool = False
    ):
        """
        Initialize the Gradio application.
        
        Args:
            db_manager: Database manager instance
            llm_engine: LLM engine instance
            semantic_engine: Optional semantic engine instance
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
        self.semantic_engine = semantic_engine
        self.theme = self.app_config.theme or gr.themes.Base()
        self.debug_mode = debug_mode
        self.app = None
        
        # Initialize history
        self.history = []
    
    def build_app(self):
        """Build the Gradio application interface."""
        with gr.Blocks(theme=self.theme, title="Text-to-SQL Interface") as self.app:
            # Header
            gr.Markdown("# Text-to-SQL Interface")
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
                                    x_column = gr.Dropdown(label="X-Axis Column")
                                    y_column = gr.Dropdown(label="Y-Axis Column")
                                    
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
                            with gr.TabItem("Debug"):
                                # Debug information
                                with gr.Accordion("Query Plan", open=True):
                                    query_plan_output = gr.JSON(label="Semantic Query Plan")
                                
                                with gr.Accordion("Execution Details", open=True):
                                    execution_details_output = gr.JSON(label="Execution Details")
                                
                                with gr.Accordion("Schema Semantics", open=False):
                                    schema_semantics_btn = gr.Button("Analyze Schema Semantics")
                                    schema_semantics_output = gr.JSON(label="Schema Semantics")
            
            # Define event handlers
            submit_btn.click(
                fn=self.handle_query,
                inputs=[query_input],
                outputs=[sql_output, status_output]
            )
            
            execute_btn.click(
                fn=self.execute_sql,
                inputs=[sql_output],
                outputs=[results_output, status_output, x_column, y_column, color_column]
            )
            
            clear_btn.click(
                fn=self.clear_results,
                inputs=[],
                outputs=[results_output, sql_output, status_output, viz_output]
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
            
            if self.debug_mode and self.semantic_engine:
                schema_semantics_btn.click(
                    fn=self.analyze_schema_semantics,
                    inputs=[],
                    outputs=[schema_semantics_output]
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
    
    def handle_query(self, query: str) -> Tuple[str, str]:
        """
        Handle a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple containing:
            - Generated SQL query
            - Status message
        """
        if not query.strip():
            return "", "Please enter a query."
        
        try:
            start_time = time.time()
            
            # Generate query plan if semantic engine is available
            query_plan = None
            if self.semantic_engine:
                query_plan = self.semantic_engine.generate_semantic_query_plan(query)
            
            # Generate SQL query
            sql_query, confidence, metadata = self.llm_engine.generate_sql_no_agents(query)
            
            # Format the status message
            generation_time = time.time() - start_time
            status = f"✅ Query processed in {generation_time:.2f} seconds with {confidence:.0%} confidence."
            
            # Add to history
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.history.append({
                "timestamp": timestamp,
                "query": query,
                "sql": sql_query,
                "confidence": confidence,
                "metadata": metadata,
                "query_plan": query_plan
            })
            
            return sql_query, status
            
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return "", f"❌ Error: {str(e)}"
    
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
            logger.error(f"Error executing query in: {e}")
            return None, f"❌ Error: {str(e)}", [], [], []
    
    def clear_results(self) -> Tuple[None, str, str, None]:
        """
        Clear the query results.
        
        Returns:
            Tuple containing empty outputs
        """
        return None, "", "Results cleared.", None
    
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
    
    def analyze_schema_semantics(self) -> Dict[str, Any]:
        """
        Analyze the semantic meaning of the database schema.
        
        Returns:
            Dictionary containing semantic information about the schema
        """
        if not self.semantic_engine:
            return {"error": "Semantic engine not available"}
        
        try:
            return self.semantic_engine.analyze_schema_semantics(refresh=True)
        except Exception as e:
            logger.error(f"Error analyzing schema semantics: {e}")
            return {"error": str(e)}