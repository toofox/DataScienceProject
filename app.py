import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from layout import homepage, frage1_layout, frage2_layout, frage3_layout, frage4_layout, frage5_layout, frage6_layout, frage7_layout

# App-Initialisierung mit Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Hauptlayout der App, das durch Scrollen die Abschnitte verbindet
app.layout = html.Div([
    # Navigationsleiste mit internen Links zu den Abschnitten
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Start", href="#start")),
            dbc.NavItem(dbc.NavLink("Frage 1", href="#frage1")),
            dbc.NavItem(dbc.NavLink("Frage 2", href="#frage2")),
            dbc.NavItem(dbc.NavLink("Frage 3", href="#frage3")),
            dbc.NavItem(dbc.NavLink("Frage 4", href="#frage4")),
            dbc.NavItem(dbc.NavLink("Frage 5", href="#frage5")),
            dbc.NavItem(dbc.NavLink("Frage 6", href="#frage6")),
            dbc.NavItem(dbc.NavLink("Frage 7", href="#frage7")),
        ],
        brand="Forschungsprojekt zu ChatGPT",
        color="dark",
        dark=True,
        className="mb-4"
    ),

    # Einfügen der Layouts für jeden Abschnitt der Seite
    homepage,
    frage1_layout,
    frage2_layout,
    frage3_layout,
    frage4_layout,
    frage5_layout,
    frage6_layout,
    frage7_layout,
])

if __name__ == '__main__':
    app.run_server(debug=True)
