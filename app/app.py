import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap'
])
server = app.server  # Needed for Render deployment

# Determine the environment
environment = os.environ.get('ENVIRONMENT', 'development')  # Default to development if not set

# Conditionally set the DB URL
if environment == 'production':
    DATABASE_URL = os.environ.get('DB_URL_PROD')
else:
    DATABASE_URL = os.environ.get('DB_URL_DEV')

engine = create_engine(DATABASE_URL)

# Function to load data from PostgreSQL
def load_data(query):
    """
    Load data from PostgreSQL database using the provided query.
    
    Args:
        query (str): SQL query to execute
    
    Returns:
        pd.DataFrame: Result of the query as a pandas DataFrame
    """
    return pd.read_sql(query, engine)

# Styling
BACKGROUND_STYLE = {
    'backgroundColor': '#f0f0f0',
    'fontFamily': 'Roboto, sans-serif',
    'padding': '20px',
}

CONTENT_STYLE = {
    'margin-left': '2rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
    'backgroundColor': 'white',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'borderRadius': '8px',
}

HEADER_STYLE = {
    'backgroundColor': '#4a4a4a',
    'padding': '1rem',
    'color': 'white',
    'borderRadius': '8px 8px 0 0',
    'marginBottom': '1rem',
}

# Define the navigation bar
def create_nav_bar(active_page):
    return html.Div([
        dcc.Link('Overview', href='/', className='nav-link', style={'color': 'white' if active_page == 'overview' else 'lightgray'}),
        dcc.Link('User Engagement', href='/engagement', className='nav-link', style={'color': 'white' if active_page == 'engagement' else 'lightgray'}),
        dcc.Link('User Experience', href='/experience', className='nav-link', style={'color': 'white' if active_page == 'experience' else 'lightgray'}),
        dcc.Link('User Satisfaction', href='/satisfaction', className='nav-link', style={'color': 'white' if active_page == 'satisfaction' else 'lightgray'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '1rem', 'backgroundColor': '#333', 'borderRadius': '0 0 8px 8px'})

# Define the layout for the overview page
overview_layout = html.Div([
    html.Div([
        html.H1("Tellco Telecom Overview Dashboard", style={'textAlign': 'center'}),
    ], style=HEADER_STYLE),
    create_nav_bar('overview'),
    html.Div([
        html.Div([
            html.H3("Top Handset Types"),
            dcc.Loading(
                id="loading-top-handsets",
                type="default",
                children=[html.Button("Generate Plot", id="btn-top-handsets"), dcc.Graph(id='top-handsets-graph')]
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H3("Top Handset Manufacturers"),
            dcc.Loading(
                id="loading-top-manufacturers",
                type="default",
                children=[html.Button("Generate Plot", id="btn-top-manufacturers"), dcc.Graph(id='top-manufacturers-graph')]
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.H3("Data Usage by Application"),
        dcc.Loading(
            id="loading-data-usage",
            type="default",
            children=[html.Button("Generate Plot", id="btn-data-usage"), dcc.Graph(id='data-usage-graph')]
        )
    ])
], style=CONTENT_STYLE)

# Define the layout for the user engagement page
engagement_layout = html.Div([
    html.Div([
        html.H1("User Engagement Dashboard", style={'textAlign': 'center'}),
    ], style=HEADER_STYLE),
    create_nav_bar('engagement'),
    html.Div([
        html.Div([
            html.H3("User Engagement Clusters"),
            dcc.Loading(
                id="loading-engagement-clusters",
                type="default",
                children=[html.Button("Generate Plot", id="btn-engagement-clusters"), dcc.Graph(id='engagement-clusters-graph')]
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H3("Top Engaged Users"),
            dcc.Loading(
                id="loading-top-engaged-users",
                type="default",
                children=[html.Button("Generate Plot", id="btn-top-engaged-users"), dcc.Graph(id='top-engaged-users-graph')]
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.H3("User Engagement Over Time"),
        dcc.Loading(
            id="loading-engagement-time",
            type="default",
            children=[html.Button("Generate Plot", id="btn-engagement-time"), dcc.Graph(id='engagement-time-graph')]
        )
    ])
], style=CONTENT_STYLE)

# Define the layout for the user experience page
experience_layout = html.Div([
    html.Div([
        html.H1("User Experience Dashboard", style={'textAlign': 'center'}),
    ], style=HEADER_STYLE),
    create_nav_bar('experience'),
    html.Div([
        html.Div([
            html.H3("Throughput by Handset Type"),
            dcc.Loading(
                id="loading-throughput",
                type="default",
                children=[html.Button("Generate Plot", id="btn-throughput"), dcc.Graph(id='throughput-graph')]
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H3("TCP Retransmission by Handset Type"),
            dcc.Loading(
                id="loading-tcp-retrans",
                type="default",
                children=[html.Button("Generate Plot", id="btn-tcp-retrans"), dcc.Graph(id='tcp-retrans-graph')]
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.H3("Network Latency Over Time"),
        dcc.Loading(
            id="loading-latency-time",
            type="default",
            children=[html.Button("Generate Plot", id="btn-latency-time"), dcc.Graph(id='latency-time-graph')]
        )
    ])
], style=CONTENT_STYLE)

# Define the layout for the user satisfaction page
satisfaction_layout = html.Div([
    html.Div([
        html.H1("User Satisfaction Dashboard", style={'textAlign': 'center'}),
    ], style=HEADER_STYLE),
    create_nav_bar('satisfaction'),
    html.Div([
        html.Div([
            html.H3("Satisfaction Score Distribution"),
            dcc.Loading(
                id="loading-satisfaction-distribution",
                type="default",
                children=[html.Button("Generate Plot", id="btn-satisfaction-distribution"), dcc.Graph(id='satisfaction-distribution-graph')]
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H3("Satisfaction vs Engagement and Experience"),
            dcc.Loading(
                id="loading-satisfaction-scatter",
                type="default",
                children=[html.Button("Generate Plot", id="btn-satisfaction-scatter"), dcc.Graph(id='satisfaction-scatter-graph')]
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.H3("Top Satisfied Customers"),
        dcc.Loading(
            id="loading-top-satisfied",
            type="default",
            children=[html.Button("Generate Table", id="btn-top-satisfied"), html.Div(id='top-satisfied-table')]
        )
    ])
], style=CONTENT_STYLE)

# Define the layout of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style=BACKGROUND_STYLE)

# Callback to update page content
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/engagement':
        return engagement_layout
    elif pathname == '/experience':
        return experience_layout
    elif pathname == '/satisfaction':
        return satisfaction_layout
    else:
        return overview_layout

# Callback for overview page graphs
@app.callback(
    [Output('top-handsets-graph', 'figure'),
     Output('top-manufacturers-graph', 'figure'),
     Output('data-usage-graph', 'figure')],
    [Input('btn-top-handsets', 'n_clicks'),
     Input('btn-top-manufacturers', 'n_clicks'),
     Input('btn-data-usage', 'n_clicks')]
)
def update_overview_graphs(btn1, btn2, btn3):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [go.Figure()] * 3
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "btn-top-handsets" or button_id == "btn-top-manufacturers":
        # Top Handset Types and Manufacturers
        query_handsets = """
        SELECT "Handset Type", COUNT(*) as count
        FROM public.xdr_data
        GROUP BY "Handset Type"
        ORDER BY count DESC
        LIMIT 10
        """
        df_handsets = load_data(query_handsets)
        fig_handsets = px.bar(df_handsets, x='Handset Type', y='count', title='Top 10 Handset Types')
        
        query_manufacturers = """
        SELECT "Handset Manufacturer", COUNT(*) as count
        FROM public.xdr_data
        GROUP BY "Handset Manufacturer"
        ORDER BY count DESC
        LIMIT 5
        """
        df_manufacturers = load_data(query_manufacturers)
        fig_manufacturers = px.pie(df_manufacturers, values='count', names='Handset Manufacturer', title='Top 5 Handset Manufacturers')
    else:
        fig_handsets = go.Figure()
        fig_manufacturers = go.Figure()
    
    if button_id == "btn-data-usage":
        # Data Usage by Application
        query_data_usage = """
        SELECT 
            SUM("Social Media DL (Bytes)" + "Social Media UL (Bytes)") as "Social Media",
            SUM("Google DL (Bytes)" + "Google UL (Bytes)") as "Google",
            SUM("Email DL (Bytes)" + "Email UL (Bytes)") as "Email",
            SUM("Youtube DL (Bytes)" + "Youtube UL (Bytes)") as "Youtube",
            SUM("Netflix DL (Bytes)" + "Netflix UL (Bytes)") as "Netflix",
            SUM("Gaming DL (Bytes)" + "Gaming UL (Bytes)") as "Gaming",
            SUM("Other DL (Bytes)" + "Other UL (Bytes)") as "Other"
        FROM public.xdr_data
        """
        df_data_usage = load_data(query_data_usage)
        df_data_usage_melted = df_data_usage.melt(var_name='Application', value_name='Data Usage')
        fig_data_usage = px.pie(df_data_usage_melted, values='Data Usage', names='Application', title='Data Usage by Application')
    else:
        fig_data_usage = go.Figure()
    
    return fig_handsets, fig_manufacturers, fig_data_usage

# Callback for user engagement page graphs
@app.callback(
    [Output('engagement-clusters-graph', 'figure'),
     Output('top-engaged-users-graph', 'figure'),
     Output('engagement-time-graph', 'figure')],
    [Input('btn-engagement-clusters', 'n_clicks'),
     Input('btn-top-engaged-users', 'n_clicks'),
     Input('btn-engagement-time', 'n_clicks')]
)
def update_engagement_graphs(btn1, btn2, btn3):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [go.Figure()] * 3
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "btn-engagement-clusters":
        query_clusters = """
        SELECT "Sessions Frequency", "Total Traffic (Bytes)", "Cluster"
        FROM public.user_clusters
        LIMIT 1000
        """
        df_clusters = load_data(query_clusters)
        fig_clusters = px.scatter(df_clusters, x='Sessions Frequency', y='Total Traffic (Bytes)', 
                                  color='Cluster', title='User Engagement Clusters')
        fig_clusters.update_layout(template='plotly_dark')
    else:
        fig_clusters = go.Figure()
    
    if button_id == "btn-top-engaged-users":
        query_top_engaged = """
        SELECT "MSISDN/Number", "Engagement Score"
        FROM public.user_satisfaction_scores
        ORDER BY "Engagement Score" DESC
        LIMIT 10
        """
        df_top_engaged = load_data(query_top_engaged)
        fig_top_engaged = px.bar(df_top_engaged, x='MSISDN/Number', y='Engagement Score', 
                                 title='Top 10 Engaged Users')
        fig_top_engaged.update_layout(template='plotly_dark')
    else:
        fig_top_engaged = go.Figure()
    
    if button_id == "btn-engagement-time":
        query_engagement_time = """
        SELECT DATE_TRUNC('day', "Start Time") as date, AVG("Engagement Score") as avg_engagement
        FROM public.user_satisfaction_scores
        GROUP BY DATE_TRUNC('day', "Start Time")
        ORDER BY date
        """
        df_engagement_time = load_data(query_engagement_time)
        fig_engagement_time = px.line(df_engagement_time, x='date', y='avg_engagement',
                                      title='Average User Engagement Over Time')
        fig_engagement_time.update_layout(template='plotly_dark')
    else:
        fig_engagement_time = go.Figure()
    
    return fig_clusters, fig_top_engaged, fig_engagement_time

# Update the engagement layout
engagement_layout = html.Div([
    html.Div([
        html.H1("User Engagement Dashboard", style={'textAlign': 'center', 'color': 'white'}),
    ], style=HEADER_STYLE),
    create_nav_bar('engagement'),
    html.Div([
        html.Div([
            html.H3("User Engagement Clusters", style={'color': 'white'}),
            dcc.Loading(
                id="loading-engagement-clusters",
                type="circle",
                children=[
                    html.Button("Generate Plot", id="btn-engagement-clusters", className='btn btn-primary'),
                    dcc.Graph(id='engagement-clusters-graph')
                ]
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3("Top Engaged Users", style={'color': 'white'}),
            dcc.Loading(
                id="loading-top-engaged-users",
                type="circle",
                children=[
                    html.Button("Generate Plot", id="btn-top-engaged-users", className='btn btn-primary'),
                    dcc.Graph(id='top-engaged-users-graph')
                ]
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]),
    html.Div([
        html.H3("User Engagement Over Time", style={'color': 'white'}),
        dcc.Loading(
            id="loading-engagement-time",
            type="circle",
            children=[
                html.Button("Generate Plot", id="btn-engagement-time", className='btn btn-primary'),
                dcc.Graph(id='engagement-time-graph')
            ]
        )
    ]),
    html.Div([
        html.H3("Top Satisfied Customers", style={'color': 'white'}),
        dcc.Loading(
            id="loading-top-satisfied",
            type="circle",
            children=[
                html.Button("Show Top Satisfied Customers", id="btn-top-satisfied", className='btn btn-primary'),
                html.Div(id='top-satisfied-table')
            ]
        )
    ])
], style=CONTENT_STYLE)

# Callback for top satisfied customers
@app.callback(
    Output('top-satisfied-table', 'children'),
    [Input('btn-top-satisfied', 'n_clicks')]
)
def update_top_satisfied(n_clicks):
    if n_clicks is None:
        return []
    
    query = """
    SELECT "MSISDN/Number", "Satisfaction Score"
    FROM public.user_satisfaction_scores
    ORDER BY "Satisfaction Score" DESC
    LIMIT 10
    """
    df = load_data(query)
    
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white',
            'textAlign': 'left'
        },
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'fontWeight': 'bold'
        }
    )

# Update the navigation bar styling
def create_nav_bar(active_page):
    return html.Div([
        dcc.Link('Overview', href='/', className='nav-link', style={'color': 'white' if active_page == 'overview' else 'lightgray', 'backgroundColor': '#007bff' if active_page == 'overview' else 'transparent'}),
        dcc.Link('User Engagement', href='/engagement', className='nav-link', style={'color': 'white' if active_page == 'engagement' else 'lightgray', 'backgroundColor': '#007bff' if active_page == 'engagement' else 'transparent'}),
        dcc.Link('User Experience', href='/experience', className='nav-link', style={'color': 'white' if active_page == 'experience' else 'lightgray', 'backgroundColor': '#007bff' if active_page == 'experience' else 'transparent'}),
        dcc.Link('User Satisfaction', href='/satisfaction', className='nav-link', style={'color': 'white' if active_page == 'satisfaction' else 'lightgray', 'backgroundColor': '#007bff' if active_page == 'satisfaction' else 'transparent'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '1rem', 'backgroundColor': '#333', 'borderRadius': '0 0 8px 8px'})

# Update the overall styling
BACKGROUND_STYLE = {
    'backgroundColor': '#1e1e1e',
    'fontFamily': 'Roboto, sans-serif',
    'padding': '20px',
    'color': 'white'
}

CONTENT_STYLE = {
    'margin-left': '2rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
    'backgroundColor': '#2c2c2c',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'borderRadius': '8px',
}

HEADER_STYLE = {
    'backgroundColor': '#4a4a4a',
    'padding': '1rem',
    'color': 'white',
    'borderRadius': '8px 8px 0 0',
    'marginBottom': '1rem',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
}

# Add this to your app.layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style=BACKGROUND_STYLE)