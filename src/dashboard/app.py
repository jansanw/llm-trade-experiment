import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json

from ..bot.trading_bot import TradingBot
from ..llm.deepseek_provider import DeepSeekProvider
from ..backtest.engine import BacktestEngine

class Dashboard:
    """Dashboard for visualizing trading bot performance."""
    
    def __init__(self, bot: TradingBot):
        """Initialize dashboard with a trading bot instance."""
        self.bot = bot
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("LLM Trading Bot Dashboard",
                           className="text-center mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Data"),
                        dbc.CardBody([
                            dcc.Graph(id="market-graph")
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Latest Decision"),
                        dbc.CardBody([
                            html.Div(id="decision-info")
                        ])
                    ]),
                    html.Br(),
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics"),
                        dbc.CardBody([
                            html.Div(id="performance-metrics")
                        ])
                    ])
                ], width=4)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Backtest Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        start_date=datetime.now() - timedelta(days=30),
                                        end_date=datetime.now(),
                                        display_format="YYYY-MM-DD"
                                    )
                                ]),
                                dbc.Col([
                                    dbc.Button(
                                        "Run Backtest",
                                        id="backtest-button",
                                        color="primary"
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            ]),
            
            dcc.Interval(
                id="update-interval",
                interval=60000,  # 1 minute in milliseconds
                n_intervals=0
            )
        ], fluid=True)
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output("market-graph", "figure"),
             Output("decision-info", "children"),
             Output("performance-metrics", "children")],
            [Input("update-interval", "n_intervals")]
        )
        async def update_live_data(n):
            # Get latest decision
            decision = await self.bot.get_trading_decision()
            
            # Get market data
            hourly_df, min15_df, min5_df, min1_df = (
                await self.bot.provider.fetch_multi_timeframe_data(
                    self.bot.symbol
                )
            )
            
            # Create market graph
            fig = self._create_market_graph(
                hourly_df,
                min15_df,
                min5_df,
                min1_df,
                decision
            )
            
            # Format decision info
            decision_info = html.Div([
                html.H4(
                    f"Position: {'LONG' if decision['position'] > 0 else 'SHORT' if decision['position'] < 0 else 'NEUTRAL'}",
                    className=f"text-{'success' if decision['position'] > 0 else 'danger' if decision['position'] < 0 else 'warning'}"
                ),
                html.H5(f"Size: {abs(decision['position']):.2f}"),
                html.H5(f"Confidence: {decision['confidence']:.2%}"),
                html.P(f"Reasoning: {decision['reasoning']}")
            ])
            
            # Calculate simple metrics
            current_price = min1_df.iloc[-1]["close"]
            day_change = (current_price / min1_df.iloc[0]["close"] - 1) * 100
            
            metrics = html.Div([
                html.H4(f"Current Price: ${current_price:.2f}"),
                html.H5(f"Day Change: {day_change:+.2f}%"),
                html.H5(f"Current Position: {self.bot.current_position:.2f}")
            ])
            
            return fig, decision_info, metrics
            
        @self.app.callback(
            [Output("market-graph", "figure"),
             Output("performance-metrics", "children")],
            [Input("backtest-button", "n_clicks")],
            [State("date-range", "start_date"),
             State("date-range", "end_date")]
        )
        async def run_backtest(n_clicks, start_date, end_date):
            if n_clicks is None:
                return dash.no_update
                
            # Run backtest
            engine = BacktestEngine(
                self.bot,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date)
            )
            results = await engine.run()
            
            # Create backtest graph
            fig = self._create_backtest_graph(results)
            
            # Format metrics
            metrics = html.Div([
                html.H4("Backtest Results"),
                html.Table([
                    html.Tr([
                        html.Td(k.replace("_", " ").title()),
                        html.Td(f"{v:.2f}")
                    ]) for k, v in results.performance_metrics.items()
                ])
            ])
            
            return fig, metrics
            
    def _create_market_graph(
        self,
        hourly_df: pd.DataFrame,
        min15_df: pd.DataFrame,
        min5_df: pd.DataFrame,
        min1_df: pd.DataFrame,
        decision: dict
    ) -> go.Figure:
        """Create market data visualization."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick charts for each timeframe
        timeframes = [
            ("1H", hourly_df),
            ("15M", min15_df),
            ("5M", min5_df),
            ("1M", min1_df)
        ]
        
        for name, df in timeframes:
            fig.add_trace(
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=name
                ),
                row=1,
                col=1
            )
            
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=min1_df["timestamp"],
                y=min1_df["volume"],
                name="Volume"
            ),
            row=2,
            col=1
        )
        
        # Add decision marker if significant
        if abs(decision["position"]) >= self.bot.position_threshold:
            fig.add_trace(
                go.Scatter(
                    x=[min1_df.iloc[-1]["timestamp"]],
                    y=[min1_df.iloc[-1]["close"]],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if decision["position"] > 0 else "triangle-down",
                        size=15,
                        color="green" if decision["position"] > 0 else "red"
                    ),
                    name=f"Decision ({decision['confidence']:.0%} conf)"
                ),
                row=1,
                col=1
            )
            
        fig.update_layout(
            title=f"{self.bot.symbol} Market Data",
            xaxis_title="Time",
            yaxis_title="Price",
            yaxis2_title="Volume",
            template="plotly_dark"
        )
        
        return fig
        
    def _create_backtest_graph(self, results: "BacktestResult") -> go.Figure:
        """Create backtest results visualization."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=results.equity_curve["timestamp"],
                y=results.equity_curve["capital"],
                name="Equity"
            ),
            row=1,
            col=1
        )
        
        # Add trade markers
        for trade in results.trades:
            fig.add_trace(
                go.Scatter(
                    x=[trade["timestamp"]],
                    y=[trade["price"]],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if trade["action"] == "BUY" else "triangle-down",
                        size=10,
                        color="green" if trade["action"] == "BUY" else "red"
                    ),
                    name=f"{trade['action']} ({trade['confidence']:.0%} conf)"
                ),
                row=1,
                col=1
            )
            
        # Add returns
        fig.add_trace(
            go.Scatter(
                x=results.equity_curve["timestamp"],
                y=results.equity_curve["returns"],
                name="Returns"
            ),
            row=2,
            col=1
        )
        
        fig.update_layout(
            title="Backtest Results",
            xaxis_title="Time",
            yaxis_title="Equity",
            yaxis2_title="Returns",
            template="plotly_dark"
        )
        
        return fig
        
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port) 