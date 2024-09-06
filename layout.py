import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
#chatgpt:
# Um  eigene CSV-Datei einzufügen, ersetzen Sie 'your_file.csv' durch den Pfad zu Ihrer CSV-Datei.
# Stellen Sie sicher, dass Ihre CSV-Datei im selben Verzeichnis wie dieses Skript liegt oder geben Sie den vollständigen Pfad an.
# df_bsp = pd.read_csv('your_file.csv')
# df_bsp.head({Anzahl der Zeilen, die angezeigt werden sollen})

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

df_asia_data = pd.DataFrame.read_csv('Data/EU_Asia_summary.csv')

# Home Section (Intro and overview)
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
        # Removed the row containing the "Explore Research" button
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

        # Research Question 5: Comparison Between CAU and Other Universities
        dbc.Row([
            dbc.Col(html.H4("Research Question 5: Comparison Between CAU and Other Universities"), width=6),
            dbc.Col(html.P(
                "What are the differences between Christian-Albrechts-Universität zu Kiel (CAU) and other German universities regarding the perceived influence of ChatGPT? "
                "This project compares the language changes at CAU with other institutions.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="comparison-universities-graph",
                figure=px.bar(df_word_usage, x="Year", y="Word_Count", color="Word_Type",
                              title="Comparison of Word Usage Across Universities")
            )),
        ], className="mb-5"),

        # Research Question 6: Faculty Differences at CAU
        dbc.Row([
            dbc.Col(html.H4("Research Question 6: Faculty Differences at CAU"), width=6),
            dbc.Col(html.P(
                "How has word usage changed across different faculties at CAU since the introduction of ChatGPT? "
                "This project analyzes differences between disciplines, such as natural sciences and humanities.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="faculty-differences-graph",
                figure=px.bar(df_word_usage, x="Year", y="Word_Count", color="Word_Type",
                              title="Comparison of Word Usage by Faculty")
            )),
        ], className="mb-5"),

        # Research Question 7: Global Comparison Between Universities
        dbc.Row([
            dbc.Col(html.H4("Research Question 7: Global Comparison Between Universities"), width=6),
            dbc.Col(html.P(
                "How do the effects of ChatGPT on scientific papers differ between various universities worldwide? "
                "This includes comparisons between top German universities, European, and Asian institutions.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="global-comparison-graph",
                figure=px.line(df_asia_data, x="PubDate", y="flag_count", color="Region", title="Comparison of EU vs Asia.")
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
                "Thank you for exploring our Page\n"
                "This Date was presented by Louis Krückmeyer, Matheus Kolzarek and Tom Skrzynski-Fox.\n"
                "This project was created to explore the impact of ChatGPT on scientific papers. \n"
                "We hope you enjoyed the analysis and visualizations presented here. \n"
                "Thank you for the great support and feedback Mirjam Bayer"
            )),
        ], className="text-center"),
    ], className="py-5")
])
