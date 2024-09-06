import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Test data
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [10, 15, 7, 20]
})
asia_df = pd.DataFrame({
    'Year': [2020, 2021, 2022],
    'KeywordCount': [100, 150, 200]
})

# Homepage layout
homepage = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Welcome to My Research Project")),
        dbc.Col(html.P("This website presents interactive visualizations and insights on the research questions I'm working on.")),
    ], className="mt-4")
])

# Layout for Research Question 1
page1_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Research Question 1: Example Analysis"))),
    dbc.Row(dbc.Col(html.P("This section provides an interactive visualization answering the first research question."))),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id="rq1-graph",
            figure=px.bar(df, x="Category", y="Values", title="Category Values")
        )),
    ]),
    dbc.Row(dbc.Col(html.P("Description: This chart visualizes the data associated with Research Question 1."))),
])

# Layout for Research Question 2
page2_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Research Question 2: Example Analysis"))),
    dbc.Row(dbc.Col(html.P("This section provides an interactive visualization answering the second research question."))),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id="rq2-graph",
            figure=px.pie(df, names="Category", values="Values", title="Category Distribution")
        )),
    ]),
    dbc.Row(dbc.Col(html.P("Description: This chart visualizes the data associated with Research Question 2."))),
])

# Layout for Research Question 3 (Asia Data)
page3_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Research Question 3: Asia Time/Keyword Analysis"))),
    dbc.Row(dbc.Col(html.P("This section provides an interactive visualization answering the third research question."))),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id="rq3-graph",
            figure=px.line(asia_df, x="Year", y="KeywordCount", title="Keyword Count Over Years in Asia")
        )),
    ]),
    dbc.Row(dbc.Col(html.P("Description: This chart visualizes the keyword count over the years in Asia."))),
])

# Layout for the Imprint page
imprint_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Contact"))),
    dbc.Row(dbc.Col(html.P("Author: [Louis Krückmeyer, Matheus Kolzarek, Tom Skrzynski-Fox]"))),
    dbc.Row(dbc.Col(html.P("Group: [Joule im Pool]"))),
])
