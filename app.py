import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html, Dash
from dash.dependencies import Input, Output
import plotly.express as px

app = Dash(__name__)
server = app.server

# IMPORT DATA

test_df = pd.read_csv('Data/Uni_Kiel_arxiv_by_name.csv')

# ADDITIONAL LAYOUT

# Navigational Bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Homepage", href="/")),
        dbc.NavItem(dbc.NavLink("Research Question 1", href="/rq1")),
        dbc.NavItem(dbc.NavLink("Research Question 2", href="/rq2")),
        dbc.NavItem(dbc.NavLink("Imprint", href="/imprint")),
    ],
    brand="Data Science Project",
    color="primary",
    dark=True,
)

homepage = html.Div([
    html.H2("Welcome to My Research Project"),
    html.P("This website presents interactive visualizations and insights on the research questions I'm working on."),
])

page1_layout = html.Div([
    html.H2("Research Question 1: Example Analysis"),
    html.P("This section provides an interactive visualization answering the first research question."),
    dcc.Graph(
        id="rq1-graph",
        figure=px.bar(test_df, x="Category", y="Values", title="Category Values")
    ),
    html.P("Description: This chart visualizes the data associated with Research Question 1."),
])

page2_layout = html.Div([
    html.H2("Research Question 2: Example Analysis"),
    html.P("This section provides an interactive visualization answering the second research question."),
    dcc.Graph(
        id="rq2-graph",
        figure=px.pie(test_df, names="Category", values="Values", title="Category Distribution")
    ),
    html.P("Description: This chart visualizes the data associated with Research Question 2."),
])

imprint_layout = html.Div([
    html.H2("Contact"),
    html.P("Author: [Louis Krückmeyer, Matheus Kolzarek, Tom Skrzynski-Fox]"),
    html.P("Group: [Joule im Pool]"),
])

# MAIN LAYOUT

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content", className="container"),
])


# CALLBACKS

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/rq1":
        return page1_layout
    elif pathname == "/rq2":
        return page2_layout
    elif pathname == "/imprint":
        return imprint_layout
    else:
        return homepage

if __name__ == '__main__':
 app.run_server(debug= True)