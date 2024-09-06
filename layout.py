import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Beispiel-Daten
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [10, 15, 7, 20]
})
asia_df = pd.DataFrame({
    'Year': [2020, 2021, 2022],
    'KeywordCount': [100, 150, 200]
})

# Bereits bestehende Layouts (Homepage, Frage 1-3)

# Layout für Forschungsfrage 4: Unterschiede zwischen Abstracts und vollständigen Texten
page4_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Frage 4: Unterschiede zwischen Abstracts und vollständigen Texten", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Untersuchung, ob es Unterschiede in der sprachlichen Gestaltung zwischen Abstracts und den vollständigen Texten wissenschaftlicher Arbeiten gibt."))),
    # Hier könnten interaktive Visualisierungen oder Analysen eingefügt werden
])

# Layout für Forschungsfrage 5: Vergleich CAU vs. andere Universitäten
page5_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Frage 5: Vergleich zwischen CAU und anderen deutschen Universitäten", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Vergleich der vermeintlichen Beeinflussung durch ChatGPT zwischen der Christian-Albrechts-Universität zu Kiel (CAU) und anderen deutschen Universitäten."))),
    # Weitere Analysen oder Visualisierungen können hier eingefügt werden
])

# Layout für Forschungsfrage 6: Vergleich zwischen Fakultäten der CAU
page6_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Frage 6: Vergleich der Wortwahl zwischen Fakultäten der CAU", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Untersuchung der Veränderungen in der Wortwahl zwischen verschiedenen Fakultäten der CAU seit der Einführung von ChatGPT."))),
    # Hier könnten Visualisierungen zum Vergleich der Fakultäten ergänzt werden
])

# Layout für Forschungsfrage 7: Vergleich globaler Universitäten
page7_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Frage 7: Unterschiede zwischen Universitäten weltweit", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Unterschiede zwischen verschiedenen Universitäten weltweit hinsichtlich der vermeintlichen Beeinflussung durch ChatGPT."))),
    # Hier könnten Vergleichsanalysen zwischen globalen Universitäten hinzugefügt werden
])

# Layout für den Imprint-Bereich
imprint_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Kontakt", style={"color": "#9b0a7d"}))),
    dbc.Row(dbc.Col(html.P("Autor: [Louis Krückmeyer, Matheus Kolzarek, Tom Skrzynski-Fox]"))),
    dbc.Row(dbc.Col(html.P("Gruppe: [Joule im Pool]"))),
])
