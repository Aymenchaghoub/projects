"""
Sales Dashboard Application

An interactive web dashboard for sales data analysis using Dash and Plotly.

Author: Chaghoub Aymen
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class SalesDashboard:
    """
    Interactive sales dashboard application.
    
    This class creates a web-based dashboard for analyzing sales data
    with interactive filters and visualizations.
    """
    
    def __init__(self, data_path: str = 'sales_data_sample.csv'):
        """
        Initialize the SalesDashboard.
        
        Args:
            data_path: Path to the sales data CSV file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.app = Dash(__name__)
        self.app.title = "Sales Dashboard"
        self._load_data()
        self._create_layout()
        self._register_callbacks()
    
    def _load_data(self) -> None:
        """
        Load and preprocess sales data.
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.data_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise ValueError("Could not read file with any encoding")
            
            print(f"✅ Data loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            
            # Normalize column names
            self.df.columns = [col.strip().upper() for col in self.df.columns]
            
            # Select relevant columns
            required_cols = ['ORDERDATE', 'COUNTRY', 'PRODUCTLINE', 'SALES', 'QUANTITYORDERED']
            available_cols = [col for col in required_cols if col in self.df.columns]
            self.df = self.df[available_cols].copy()
            
            # Convert data types
            self.df['ORDERDATE'] = pd.to_datetime(self.df.get('ORDERDATE'), errors='coerce')
            self.df['SALES'] = pd.to_numeric(self.df.get('SALES'), errors='coerce')
            self.df['QUANTITYORDERED'] = pd.to_numeric(self.df.get('QUANTITYORDERED'), errors='coerce')
            
            # Calculate profit if not present
            if 'PROFIT' not in self.df.columns:
                self.df['PROFIT'] = self.df['SALES'] * 0.1
            else:
                self.df['PROFIT'] = pd.to_numeric(self.df.get('PROFIT'), errors='coerce')
            
            # Remove rows with missing critical data
            self.df = self.df.dropna(subset=['ORDERDATE', 'SALES'])
            
            print(f"✅ Data processed: {self.df.shape[0]} valid records")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Create sample data if file not found
            self._create_sample_data()
    
    def _create_sample_data(self) -> None:
        """
        Create sample data for demonstration purposes.
        """
        print("📊 Creating sample data...")
        
        import numpy as np
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        n_records = 1000
        
        # Generate sample data
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia']
        products = ['Motorcycles', 'Classic Cars', 'Trucks and Buses', 'Vintage Cars']
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        self.df = pd.DataFrame({
            'ORDERDATE': pd.date_range(start_date, end_date, periods=n_records),
            'COUNTRY': np.random.choice(countries, n_records),
            'PRODUCTLINE': np.random.choice(products, n_records),
            'SALES': np.random.uniform(1000, 50000, n_records),
            'QUANTITYORDERED': np.random.randint(1, 100, n_records)
        })
        
        self.df['PROFIT'] = self.df['SALES'] * np.random.uniform(0.05, 0.25, n_records)
        
        print(f"✅ Sample data created: {self.df.shape[0]} records")
    
    def _create_layout(self) -> None:
        """
        Create the dashboard layout.
        """
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("📊 Sales Dashboard", 
                       style={"textAlign": "center", "marginBottom": "20px"}),
                html.P("Interactive sales analytics and performance metrics",
                      style={"textAlign": "center", "color": "gray", "marginBottom": "30px"})
            ]),
            
            # Error section
            html.Div(id="error-section"),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Region:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="region-filter",
                        options=[{"label": "All", "value": "All"}] + 
                               [{"label": region, "value": region} 
                                for region in sorted(self.df['COUNTRY'].dropna().unique())],
                        value="All",
                        style={"marginBottom": "10px"}
                    )
                ], style={"width": "45%", "display": "inline-block", "marginRight": "5%"}),
                
                html.Div([
                    html.Label("Product:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="product-filter",
                        options=[{"label": "All", "value": "All"}] + 
                               [{"label": product, "value": product} 
                                for product in sorted(self.df['PRODUCTLINE'].dropna().unique())],
                        value="All",
                        style={"marginBottom": "10px"}
                    )
                ], style={"width": "45%", "display": "inline-block"})
            ], style={"marginBottom": "30px", "padding": "20px", 
                     "backgroundColor": "#f8f9fa", "borderRadius": "10px"}),
            
            # KPIs
            html.Div(id="kpi-section", 
                    style={"display": "flex", "justifyContent": "space-around", 
                           "marginBottom": "40px"}),
            
            # Charts
            html.Div([
                html.Div([
                    dcc.Graph(id="sales-trend")
                ], style={"width": "48%", "display": "inline-block", "marginRight": "2%"}),
                
                html.Div([
                    dcc.Graph(id="sales-region")
                ], style={"width": "48%", "display": "inline-block"})
            ], style={"marginBottom": "30px"}),
            
            html.Div([
                html.H3("Top 5 Products", style={"textAlign": "center"}),
                dcc.Graph(id="top-products")
            ], style={"marginBottom": "30px"}),
            
            html.Div([
                html.H3("Units Sold by Region", style={"textAlign": "center"}),
                dcc.Graph(id="units-region")
            ])
        ], style={"padding": "20px", "fontFamily": "Arial, sans-serif"})
    
    def _register_callbacks(self) -> None:
        """
        Register dashboard callbacks.
        """
        @self.app.callback(
            [Output("error-section", "children"),
             Output("kpi-section", "children"),
             Output("sales-trend", "figure"),
             Output("sales-region", "figure"),
             Output("top-products", "figure"),
             Output("units-region", "figure")],
            [Input("region-filter", "value"),
             Input("product-filter", "value")]
        )
        def update_dashboard(selected_region: str, selected_product: str) -> Tuple:
            """
            Update dashboard components based on filter selections.
            
            Args:
                selected_region: Selected region filter
                selected_product: Selected product filter
                
            Returns:
                Tuple of updated components
            """
            try:
                # Filter data
                filtered_df = self.df.copy()
                
                if selected_region != "All":
                    filtered_df = filtered_df[filtered_df['COUNTRY'] == selected_region]
                
                if selected_product != "All":
                    filtered_df = filtered_df[filtered_df['PRODUCTLINE'] == selected_product]
                
                # Calculate KPIs
                total_sales = filtered_df['SALES'].sum()
                total_profit = filtered_df['PROFIT'].sum()
                total_units = filtered_df['QUANTITYORDERED'].sum()
                profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
                
                # Create KPI cards
                kpi_cards = [
                    html.Div([
                        html.H4("Total Sales", style={"margin": "0", "fontSize": "14px"}),
                        html.H2(f"${total_sales:,.0f}", 
                               style={"margin": "0", "color": "#6a1b9a", "fontSize": "24px"})
                    ], style={"textAlign": "center", "padding": "20px", 
                             "backgroundColor": "white", "borderRadius": "10px",
                             "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                    
                    html.Div([
                        html.H4("Total Profit", style={"margin": "0", "fontSize": "14px"}),
                        html.H2(f"${total_profit:,.0f}", 
                               style={"margin": "0", "color": "#4caf50", "fontSize": "24px"})
                    ], style={"textAlign": "center", "padding": "20px", 
                             "backgroundColor": "white", "borderRadius": "10px",
                             "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                    
                    html.Div([
                        html.H4("Units Sold", style={"margin": "0", "fontSize": "14px"}),
                        html.H2(f"{total_units:,.0f}", 
                               style={"margin": "0", "color": "#ff9800", "fontSize": "24px"})
                    ], style={"textAlign": "center", "padding": "20px", 
                             "backgroundColor": "white", "borderRadius": "10px",
                             "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                    
                    html.Div([
                        html.H4("Profit Margin", style={"margin": "0", "fontSize": "14px"}),
                        html.H2(f"{profit_margin:.1f}%", 
                               style={"margin": "0", "color": "#2196f3", "fontSize": "24px"})
                    ], style={"textAlign": "center", "padding": "20px", 
                             "backgroundColor": "white", "borderRadius": "10px",
                             "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})
                ]
                
                # Create charts
                fig_trend = self._create_trend_chart(filtered_df)
                fig_region = self._create_region_chart(filtered_df)
                fig_products = self._create_products_chart(filtered_df)
                fig_units = self._create_units_chart(filtered_df)
                
                return "", kpi_cards, fig_trend, fig_region, fig_products, fig_units
                
            except Exception as e:
                error_msg = html.Div([
                    html.H2("Error", style={"color": "red", "textAlign": "center"}),
                    html.P(str(e), style={"textAlign": "center"})
                ])
                return error_msg, [], {}, {}, {}, {}
    
    def _create_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create sales trend chart.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            Plotly figure
        """
        # Group by month
        df_monthly = df.groupby(df['ORDERDATE'].dt.to_period("M")).agg({
            'SALES': 'sum',
            'PROFIT': 'sum'
        }).reset_index()
        
        df_monthly['ORDERDATE'] = df_monthly['ORDERDATE'].dt.to_timestamp()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_monthly['ORDERDATE'],
            y=df_monthly['SALES'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_monthly['ORDERDATE'],
            y=df_monthly['PROFIT'],
            mode='lines+markers',
            name='Profit',
            line=dict(color='#10b981', width=3),
            marker=dict(size=6),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Sales and Profit Trends",
            xaxis_title="Month",
            yaxis_title="Sales ($)",
            yaxis2=dict(title="Profit ($)", overlaying="y", side="right"),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_region_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create sales by region pie chart.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            Plotly figure
        """
        region_sales = df.groupby('COUNTRY')['SALES'].sum().reset_index()
        
        fig = px.pie(
            region_sales,
            values='SALES',
            names='COUNTRY',
            title="Sales Distribution by Region",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        
        return fig
    
    def _create_products_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create top products chart.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            Plotly figure
        """
        top_products = df.groupby('PRODUCTLINE')[['SALES', 'QUANTITYORDERED']].sum().reset_index()
        top_products = top_products.sort_values('SALES', ascending=False).head(5)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Sales',
            x=top_products['PRODUCTLINE'],
            y=top_products['SALES'],
            marker_color='#3b82f6'
        ))
        
        fig.add_trace(go.Bar(
            name='Units Sold',
            x=top_products['PRODUCTLINE'],
            y=top_products['QUANTITYORDERED'],
            marker_color='#10b981',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Top 5 Products",
            xaxis_title="Product",
            yaxis_title="Sales ($)",
            yaxis2=dict(title="Units Sold", overlaying="y", side="right"),
            barmode='group',
            template='plotly_white'
        )
        
        return fig
    
    def _create_units_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create units sold by region chart.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            Plotly figure
        """
        units_by_region = df.groupby('COUNTRY')['QUANTITYORDERED'].sum().reset_index()
        
        fig = px.pie(
            units_by_region,
            values='QUANTITYORDERED',
            names='COUNTRY',
            title="Units Sold by Region",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        
        return fig
    
    def run(self, debug: bool = True, port: int = 8050) -> None:
        """
        Run the dashboard application.
        
        Args:
            debug: Enable debug mode
            port: Port to run the application on
        """
        print(f"🚀 Starting Sales Dashboard on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)


def main():
    """
    Main function to run the sales dashboard.
    """
    dashboard = SalesDashboard('sales_data_sample.csv')
    dashboard.run()


if __name__ == "__main__":
    main()
