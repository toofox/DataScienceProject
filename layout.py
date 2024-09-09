import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
#chatgpt:
# Um  eigene CSV-Datei einzufügen, ersetzen Sie 'your_file.csv' durch den Pfad zu Ihrer CSV-Datei.
# Stellen Sie sicher, dass Ihre CSV-Datei im selben Verzeichnis wie dieses Skript liegt oder geben Sie den vollständigen Pfad an.
# df_bsp = pd.read_csv('your_file.csv')
# df_bsp.head({Anzahl der Zeilen, die angezeigt werden sollen})





#DATEN AUS CSV
#Frage 7.1
colors_blind_friendly = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
df_universities = pd.read_csv('Data/Merged_Germany_datasets.csv')
df_fachhochschulen = pd.read_csv('Data/Merged_FH_datasets.csv')
df_universities['type'] = 'Universität'
df_fachhochschulen['type'] = 'Fachhochschule'
df_combined = pd.concat([df_universities, df_fachhochschulen])
df_combined['flag'] = df_combined['flag'].apply(lambda x: 'Flagged' if x == 'Yes' else 'Not Flagged')

# Neue Spalte, die angibt, ob das Paper vor oder nach dem 1.1.2023 publiziert wurde
df_combined['DateCategory'] = df_combined['PubDate'].apply(lambda x: 'Before 1.1.2023' if x < 2023 else 'After 1.1.2023')

# Aggregation: Anzahl der markierten Artikel pro Kategorie (Universität oder Fachhochschule)
df_grouped = df_combined.groupby(['type', 'flag']).size().reset_index(name='count')

# Berechnung der Gesamtzahl der Paper pro Typ (Universität/Fachhochschule)
df_total = df_combined.groupby('type').size().reset_index(name='total_count')

# Verknüpfe die Gesamtzahlen mit den gruppierten Daten
df_grouped = pd.merge(df_grouped, df_total, on='type')

# Berechne den Prozentsatz der markierten Artikel
df_grouped['percentage'] = (df_grouped['count'] / df_grouped['total_count']) * 100

# Filtere nur die markierten Artikel (Flagged)
df_flagged = df_grouped[df_grouped['flag'] == 'Flagged']

# Aggregation nach der Publikationsdatum-Kategorie (vor und nach 1.1.2023) für markierte Artikel
df_date_grouped = df_combined[df_combined['flag'] == 'Flagged'].groupby(['type', 'DateCategory']).size().reset_index(name='count')

# Berechnung der Gesamtzahl der Paper nach Publikationsdatum für den Prozentsatz
df_total_date = df_combined.groupby(['type', 'DateCategory']).size().reset_index(name='total_count')

# Verknüpfen der Gesamtzahlen mit den gruppierten Daten
df_date_grouped = pd.merge(df_date_grouped, df_total_date, on=['type', 'DateCategory'])

# Berechnung des Prozentsatzes
df_date_grouped['percentage'] = (df_date_grouped['count'] / df_date_grouped['total_count']) * 100
# Filter für die Jahre 2021-2024
df_filtered_years = df_combined[df_combined['PubDate'].isin([2017,2018,2019,2020, 2021, 2022, 2023, 2024])]

# Aggregation der markierten Paper nach Jahr und Institutionstyp
df_year_grouped = df_filtered_years[df_filtered_years['flag'] == 'Flagged'].groupby(['type', 'PubDate']).size().reset_index(name='count')

# Berechnung der Gesamtzahl der Paper pro Jahr für den Prozentsatz
df_total_years = df_filtered_years.groupby(['type', 'PubDate']).size().reset_index(name='total_count')

# Verknüpfen der Gesamtzahlen mit den gruppierten Daten
df_year_grouped = pd.merge(df_year_grouped, df_total_years, on=['type', 'PubDate'])

# Berechnung des Prozentsatzes
df_year_grouped['percentage'] = (df_year_grouped['count'] / df_year_grouped['total_count']) * 100







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

        # Research Question 7: Comparison Between Universities and Fachhochschulen (Percentage of Flagged Papers)
        dbc.Row([
            dbc.Col(html.H4("Research Question 7: Global Comparison Between Universities and Fachhochschulen"),
                    width=6),
            dbc.Col(html.P(
                "How do the effects of ChatGPT on scientific papers differ between various universities worldwide? "
                "This includes comparisons between top German universities and Fachhochschulen.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="comparison-graph",
                figure=px.bar(df_flagged, x='type', y='percentage', color='type',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers between Universities and Fachhochschulen",
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers'})
            )),
        ], className="mb-5"),

        # Comparison Before and After 1.1.2023
        dbc.Row([
            dbc.Col(html.H4("Research Question: Comparison of Papers Before and After 1.1.2023"), width=6),
            dbc.Col(html.P(
                "How has the percentage of flagged papers changed before and after 1.1.2023? "
                "This graph shows a comparison of flagged papers before and after the introduction of ChatGPT.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="date-comparison-graph",
                figure=px.bar(df_date_grouped, x='type', y='percentage', color='DateCategory',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers Before and After 1.1.2023",
                              barmode='group',
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers',
                                      'DateCategory': 'Publication Date Category'})
            )),
        ], className="mb-5"),

        # Line graph for 2020-2024 flagged papers
        dbc.Row([
            dbc.Col(html.H4("Research Question: Flagged Papers from 2017 to 2024"), width=6),
            dbc.Col(html.P(
                "This graph shows the percentage of flagged papers from 2017 to 2024, separated by institution type.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="year-comparison-graph",
                figure=px.line(df_year_grouped, x='PubDate', y='percentage', color='type',
                               color_discrete_sequence=colors_blind_friendly,
                               title="Percentage of Flagged Papers from 2017 to 2024, separated by institution type.Frage ",
                               labels={'PubDate': 'Publication Year', 'percentage': 'Percentage of Papers',
                                       'type': 'Institution Type'})
                .update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Setzt nur ganze Zahlen als X-Achsen-Werte
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
