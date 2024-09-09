import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Dash app
app = dash.Dash(__name__)
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
def load_data(metric, top_n):
    if metric == 'throughput':
        query = f"""
        SELECT "Handset Type", "Avg Bearer TP DL (kbps)"
        FROM public.xdr_data
        WHERE "Handset Type" IN (
            SELECT "Handset Type"
            FROM public.xdr_data
            GROUP BY "Handset Type"
            ORDER BY COUNT(*) DESC
            LIMIT {top_n}
        )
        """
    elif metric == 'tcp_retrans':
        query = f"""
        SELECT "Handset Type", "TCP DL Retrans. Vol (Bytes)"
        FROM public.xdr_data
        WHERE "Handset Type" IN (
            SELECT "Handset Type"
            FROM public.xdr_data
            GROUP BY "Handset Type"
            ORDER BY COUNT(*) DESC
            LIMIT {top_n}
        )
        """
    elif metric == 'eng_cluster':
        query = f"""
        SELECT "Dur.(s)", "Total DL (Bytes)","Total UL (Bytes)","MSISDN/Number"
        FROM public.xdr_data
        GROUP BY "MSISDN/Number"
      
        """
    else:
        print("do nothing")
    df = pd.read_sql(query, engine)
    return df

# Define the layout of the app
app.layout = html.Div([
    html.H1("Telecom User Experience Dashboard"),
    
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Throughput', 'value': 'throughput'},
            {'label': 'TCP Retransmission', 'value': 'tcp_retrans'},
            {'label': 'User Engagement Cluster', 'value': 'eng_cluster'}
        ],
        value='throughput',
        style={'width': '50%'}
    ),
    
    dcc.Slider(
        id='top-n-slider',
        min=5,
        max=20,
        step=1,
        value=10,
        marks={i: str(i) for i in range(5, 21, 5)},
    ),
    
    html.Button('Generate Graph', id='generate-button', n_clicks=0),
    
    dcc.Graph(id='main-graph')
])

# Callback to update the graph
@app.callback(
    Output('main-graph', 'figure'),
    [Input('generate-button', 'n_clicks')],
    [State('metric-dropdown', 'value'),
     State('top-n-slider', 'value')]
)
def update_graph(n_clicks, selected_metric, top_n):
    if n_clicks == 0:
        return {}  # Return empty figure on initial load
    
    df = load_data(selected_metric, top_n)
    
    if selected_metric == 'throughput':
        fig = px.box(df, x='Handset Type', y='Avg Bearer TP DL (kbps)', 
                     title=f'Distribution of Average Downlink Throughput for Top {top_n} Handset Types')
    else:  # TCP Retransmission
        fig = px.bar(df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().reset_index(), 
                     x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', 
                     title=f'Average Downlink TCP Retransmission Volume for Top {top_n} Handset Types')
    
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)