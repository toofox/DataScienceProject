import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from layout import homepage, page1_layout, page2_layout, page3_layout, imprint_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Main layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Homepage", href="/")),
            dbc.NavItem(dbc.NavLink("Research Question 1", href="/rq1")),
            dbc.NavItem(dbc.NavLink("Research Question 2", href="/rq2")),
            dbc.NavItem(dbc.NavLink("Asia Data", href="/rq3")),
            dbc.NavItem(dbc.NavLink("Imprint", href="/imprint")),
        ],
        brand="Data Science Project",
        color="primary",
        dark=True,
    ),
    html.Div(id="page-content", className="container-fluid"),
])

# Callback to manage page routing
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/rq1":
        return page1_layout
    elif pathname == "/rq2":
        return page2_layout
    elif pathname == "/rq3":
        return page3_layout
    elif pathname == "/imprint":
        return imprint_layout
    else:
        return homepage

if __name__ == '__main__':
    app.run_server(debug=True)
