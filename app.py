import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from layout import homepage, page1_layout, page2_layout, page3_layout, page4_layout, page5_layout, page6_layout, page7_layout, imprint_layout

# Verwende Bootstrap Lux Theme für ein moderneres Aussehen
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Layout mit einer eleganten und nicht blauen Navigationsleiste, die die Forschungsfragen anzeigt
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dbc.Navbar(
        dbc.Container([
            dbc.Row(
                dbc.Col(html.H3("Forschungsfragen zu ChatGPT in wissenschaftlichen Arbeiten", style={"color": "#9b0a7d"})),
                align="center",
            ),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Homepage", href="/")),
                dbc.NavItem(dbc.NavLink("Frage 1: Wortverwendung", href="/rq1")),
                dbc.NavItem(dbc.NavLink("Frage 2: Fragewörter", href="/rq2")),
                dbc.NavItem(dbc.NavLink("Frage 3: Satzlängen", href="/rq3")),
                dbc.NavItem(dbc.NavLink("Frage 4: Abstract vs. Volltext", href="/rq4")),
                dbc.NavItem(dbc.NavLink("Frage 5: CAU vs. andere Universitäten", href="/rq5")),
                dbc.NavItem(dbc.NavLink("Frage 6: Fakultäten der CAU", href="/rq6")),
                dbc.NavItem(dbc.NavLink("Frage 7: Globale Universitäten", href="/rq7")),
                dbc.NavItem(dbc.NavLink("Imprint", href="/imprint")),
            ], pills=True)
        ]),
        color="dark",  # Dunkles Design
        dark=True,  # Helle Schrift auf dunklem Hintergrund
        className="mb-4"
    ),
    html.Div(id="page-content", className="container-fluid"),
])

# Callback zur Steuerung der Navigation
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/rq1":
        return page1_layout
    elif pathname == "/rq2":
        return page2_layout
    elif pathname == "/rq3":
        return page3_layout
    elif pathname == "/rq4":
        return page4_layout
    elif pathname == "/rq5":
        return page5_layout
    elif pathname == "/rq6":
        return page6_layout
    elif pathname == "/rq7":
        return page7_layout
    elif pathname == "/imprint":
        return imprint_layout
    else:
        return homepage

if __name__ == '__main__':
    app.run_server(debug=True)
