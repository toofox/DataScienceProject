import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Example CSV data
df_word_usage = pd.DataFrame({
    'Year': ['2020', '2021', '2022'],
    'Word_Count': [100, 150, 200],
    'Word_Type': ['Technical', 'Non-Technical', 'Mixed']
})

df_sentence_length = pd.DataFrame({
    'Year': ['2020', '2021', '2022'],
    'Avg_Sentence_Length': [12.5, 13.0, 14.0]
})

df_asia_data = pd.DataFrame({
    'Year': ['2020', '2021', '2022'],
    'Keyword_Count': [1200, 1300, 1500]
})

# Home Section (Intro and overview)
homepage = html.Div(id="start", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to our Data Science Project!", style={"color": "#9b0a7d"}), className="mt-5"),
        ]),
        dbc.Row([
            dbc.Col(html.P(
                "Below, you'll find a series of research questions exploring the impact of ChatGPT on scientific papers."
            )),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Button("Explore Research", color="primary", style={"background-color": "#9b0a7d"},
                           href="#projects"),
            ),
        ], className="pt-4 text-center"),
    ], className="py-5")
])

# Projects Section now contains research questions and actual data visualizations
projects_section = html.Div(id="projects", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Research Questions", style={"color": "#9b0a7d"}), className="mt-5 mb-4 text-center"),
        ]),

        # Research Question 1: Changes in Word Usage (using real data)
        dbc.Row([
            dbc.Col(html.H4("Research Question 1: Changes in Word Usage in Scientific Papers"), width=6),
            dbc.Col(html.P(
                "How has the usage of specific words in scientific papers changed since the introduction of ChatGPT? "
                "This analysis aims to identify whether certain terms have become more or less frequent, potentially due to the influence of ChatGPT.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="word-usage-graph",
                figure=px.bar(df_word_usage, x="Year", y="Word_Count", color="Word_Type",
                              title="Word Usage Over Time by Type")
            )),
        ], className="mb-5"),

        # Research Question 2: Changes in the Use of Question Words
        dbc.Row([
            dbc.Col(html.H4("Research Question 2: Changes in the Use of Question Words"), width=6),
            dbc.Col(html.P(
                "What changes are there in the use of question words (e.g., what, why) in scientific papers since the introduction of ChatGPT? "
                "This section explores the frequency of question words before and after 2022.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="question-words-graph",
                figure=px.pie(df_word_usage, names="Word_Type", values="Word_Count",
                              title="Distribution of Question Words")
            )),
        ], className="mb-5"),

        # Research Question 3: Sentence Length in Scientific Papers
        dbc.Row([
            dbc.Col(html.H4("Research Question 3: Sentence Length in Scientific Papers"), width=6),
            dbc.Col(html.P(
                "How has the length of sentences in scientific papers changed since the introduction of ChatGPT? "
                "We aim to investigate whether sentences have become longer or shorter post-ChatGPT.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="sentence-length-graph",
                figure=px.line(df_sentence_length, x="Year", y="Avg_Sentence_Length",
                               title="Average Sentence Length Over Time")
            )),
        ], className="mb-5"),

        # Research Question 4: Asia University Keyword Analysis
        dbc.Row([
            dbc.Col(html.H4("Research Question 4: Keyword Frequency in Asia Universities"), width=6),
            dbc.Col(html.P(
                "This analysis explores how keyword usage has changed in Asian universities since the introduction of ChatGPT.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="asia-keyword-graph",
                figure=px.line(df_asia_data, x="Year", y="Keyword_Count",
                               title="Keyword Count in Asian Universities Over Time")
            )),
        ], className="mb-5"),
    ], className="py-5")
])

# About Section
about_section = html.Div(id="about", children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("About Me", style={"color": "#9b0a7d"}), className="mt-5 mb-4 text-center"),
        ]),
        dbc.Row([
            dbc.Col(html.P(
                "Thank you for exploring our Page"
                "This Date was presented by Louis Krückmeyer, Matheus Kolzarek and Tom Skrzynski-Fox."
                "This project was created to explore the impact of ChatGPT on scientific papers. "
                "We hope you enjoyed the analysis and visualizations presented here. "
                "Thank you for the great support and feedback Mirjam Bayer"
            )),
        ], className="text-center"),
    ], className="py-5")
])
