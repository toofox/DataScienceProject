import pandas as pd
import dash
from dash import dcc
from dash import html, Dash
from dash.dependencies import Input, Output
import plotly.express as px

app = Dash(__name__)
server = app.server

# IMPORT DATA

test_df = pd.read_csv('Data/Uni_Kiel_arxiv_by_name.csv')

# LAYOUT

app.layout = html.Div([
    html.H1('Dashboard', style={'textAlign': 'center', 'color': '#4CAF50'}),
    html.P('An interactive data science project.', style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Tab 1', children=[
            html.Div('Content of Tab 1')
        ]),
        dcc.Tab(label='Tab 2', children=[
            html.Div('Content of Tab 2')
        ])
    ])
])

# CALLBACKS

if __name__ == '__main__':
 app.run_server(debug= True)