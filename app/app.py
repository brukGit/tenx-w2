import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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
def load_data():
    query = "SELECT * FROM public.xdr_data"
    df = pd.read_sql(query, engine)
    return df

# Load initial data
df = load_data()

# Define the layout of the app
app.layout = html.Div([
    html.H1("Telecom User Experience Dashboard"),
    
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Throughput', 'value': 'throughput'},
            {'label': 'TCP Retransmission', 'value': 'tcp_retrans'}
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
    
    dcc.Graph(id='main-graph')
])

# Callback to update the graph
@app.callback(
    Output('main-graph', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('top-n-slider', 'value')]
)
def update_graph(selected_metric, top_n):
    top_handsets = df['Handset Type'].value_counts().nlargest(top_n).index.tolist()
    filtered_df = df[df['Handset Type'].isin(top_handsets)]
    
    if selected_metric == 'throughput':
        fig = px.box(filtered_df, x='Handset Type', y='Avg Bearer TP DL (kbps)', 
                     title=f'Distribution of Average Downlink Throughput for Top {top_n} Handset Types')
    else:  # TCP Retransmission
        fig = px.bar(filtered_df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().reset_index(), 
                     x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', 
                     title=f'Average Downlink TCP Retransmission Volume for Top {top_n} Handset Types')
    
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug=True, port=8050)