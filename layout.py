import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Beispiel-Daten
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [10, 15, 7, 20]
})

# Homepage Layout
homepage = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Willkommen zu unserem Forschungsprojekt", style={"color": "#9b0a7d"})),
        dbc.Col(html.P("Diese Website präsentiert interaktive Visualisierungen und Einblicke in unsere Forschungsfragen zum Einfluss von ChatGPT auf wissenschaftliche Arbeiten.")),
    ], className="mt-4"),
])

# Layout für Forschungsfrage 1: Veränderung der Wortverwendung
page1_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Frage 1: Veränderung der Wortverwendung", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Wie hat sich die Verwendung bestimmter Wörter in wissenschaftlichen Arbeiten seit der Einführung von ChatGPT verändert?"))),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id="rq1-graph",
            figure=px.bar(df, x="Category", y="Values", title="Kategoriehäufigkeiten")
        )),
    ]),
    dbc.Row(dbc.Col(html.P("Diese Grafik zeigt die Häufigkeiten bestimmter Wörter im Zeitverlauf."))),
])

# Layout für Forschungsfrage 2: Veränderung der Fragewörter
page2_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Frage 2: Veränderung der Fragewörter", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Welche Veränderungen gibt es in der Nutzung von Fragewörtern in wissenschaftlichen Arbeiten seit der Einführung von ChatGPT?"))),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id="rq2-graph",
            figure=px.pie(df, names="Category", values="Values", title="Verteilung von Fragewörtern")
        )),
    ]),
    dbc.Row(dbc.Col(html.P("Diese Visualisierung zeigt die Verteilung von Fragewörtern in wissenschaftlichen Arbeiten."))),
])

# Layout für Forschungsfrage 3: Satzlängen in wissenschaftlichen Arbeiten
page3_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Frage 3: Satzlängen in wissenschaftlichen Arbeiten", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Wie haben sich die Satzlängen in wissenschaftlichen Arbeiten seit der Einführung von ChatGPT verändert?"))),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id="rq3-graph",
            figure=px.line(df, x="Category", y="Values", title="Satzlängen im Zeitverlauf")
        )),
    ]),
    dbc.Row(dbc.Col(html.P("Diese Grafik zeigt die Veränderung der Satzlängen über die Zeit."))),
])

# Layout für die restlichen Fragen kannst du analog hinzufügen...
