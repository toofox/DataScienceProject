import dash_bootstrap_components as dbc
from dash import html, dcc

# Abschnitt "Start"
homepage = html.Div(id="start", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Willkommen zum Forschungsprojekt", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Auf dieser Seite werden verschiedene Forschungsfragen zu den Auswirkungen von ChatGPT auf wissenschaftliche Arbeiten dargestellt."))
        ]),
    ], className="pt-5")
])

# Abschnitt "Frage 1"
frage1_layout = html.Div(id="frage1", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Frage 1: Veränderung der Wortverwendung", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Wie hat sich die Verwendung bestimmter Wörter in wissenschaftlichen Arbeiten seit der Einführung von ChatGPT verändert?"))
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="frage1-graph",
                figure={  # Beispiel-Plot
                    'data': [{'x': ['2020', '2021', '2022'], 'y': [20, 35, 50], 'type': 'bar', 'name': 'Wörter'}],
                    'layout': {'title': 'Veränderung der Wortverwendung'}
                }
            )),
        ]),
    ], className="pt-5")
])

# Abschnitt "Frage 2"
frage2_layout = html.Div(id="frage2", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Frage 2: Veränderung der Fragewörter", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Welche Veränderungen gibt es in der Nutzung von Fragewörtern in wissenschaftlichen Arbeiten seit der Einführung von ChatGPT?"))
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="frage2-graph",
                figure={  # Beispiel-Plot
                    'data': [{'labels': ['Was', 'Warum', 'Wie'], 'values': [30, 20, 50], 'type': 'pie'}],
                    'layout': {'title': 'Verwendung von Fragewörtern'}
                }
            )),
        ]),
    ], className="pt-5")
])

# Abschnitt "Frage 3"
frage3_layout = html.Div(id="frage3", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Frage 3: Veränderung der Satzlängen", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Wie haben sich die Satzlängen in wissenschaftlichen Arbeiten seit der Einführung von ChatGPT verändert?"))
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="frage3-graph",
                figure={  # Beispiel-Plot
                    'data': [{'x': ['2020', '2021', '2022'], 'y': [25, 40, 55], 'type': 'line'}],
                    'layout': {'title': 'Satzlängen im Zeitverlauf'}
                }
            )),
        ]),
    ], className="pt-5")
])

# Abschnitt "Frage 4"
frage4_layout = html.Div(id="frage4", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Frage 4: Unterschiede zwischen Abstracts und Texten", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Untersuchung der Unterschiede zwischen Abstracts und vollständigen wissenschaftlichen Texten in Bezug auf die Verwendung von ChatGPT."))
        ]),
    ], className="pt-5")
])

# Abschnitt "Frage 5"
frage5_layout = html.Div(id="frage5", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Frage 5: Vergleich CAU vs. andere Universitäten", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Vergleich der Einflüsse von ChatGPT auf wissenschaftliche Arbeiten zwischen der Christian-Albrechts-Universität zu Kiel (CAU) und anderen deutschen Universitäten."))
        ]),
    ], className="pt-5")
])

# Abschnitt "Frage 6"
frage6_layout = html.Div(id="frage6", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Frage 6: Vergleich zwischen CAU-Fakultäten", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Vergleich der Wortwahl und Struktur wissenschaftlicher Arbeiten zwischen verschiedenen Fakultäten der CAU."))
        ]),
    ], className="pt-5")
])

# Abschnitt "Frage 7"
frage7_layout = html.Div(id="frage7", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Frage 7: Globale Unterschiede bei Universitäten", style={"color": "#9b0a7d"}), className="mt-4"),
            dbc.Col(html.P("Unterschiede in der Verwendung von ChatGPT in wissenschaftlichen Arbeiten an globalen Universitäten."))
        ]),
    ], className="pt-5")
])
