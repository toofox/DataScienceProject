import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
from scipy import stats
import numpy as np
from collections import Counter
import re
import dash
from scipy.stats import chi2_contingency, shapiro, norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
years_7_1 = df_year_grouped['PubDate'].values
percentages_7_1 = df_year_grouped['percentage'].values

# Perform linear regression to detect trend for Frage 7.1
slope_7_1, intercept_7_1, r_value_7_1, p_value_7_1, std_err_7_1 = stats.linregress(years_7_1, percentages_7_1)

# Add trend line to the existing line chart for Frage 7.1
df_year_grouped['trend_7_1'] = intercept_7_1 + slope_7_1 * df_year_grouped['PubDate']

# Display the p-value in the graph title for Frage 7.1
title_with_p_value_7_1 = f"Percentage of Flagged Papers for 2020-2024 (Universities vs Fachhochschulen)\nTrend Line (p-value: {p_value_7_1:.4f})"

# 7.2 EU und Asien Vergleich
df_eu = pd.read_csv('Data/Merged_EU_datasets.csv')
df_asia = pd.read_csv('Data/Merged_Asia_datasets.csv')

df_eu['type'] = 'EU'
df_asia['type'] = 'Asien'
df_combined_eu_asia = pd.concat([df_eu, df_asia])
df_combined_eu_asia['flag'] = df_combined_eu_asia['flag'].apply(lambda x: 'Flagged' if x == 'Yes' else 'Not Flagged')

df_combined_eu_asia['DateCategory'] = df_combined_eu_asia['PubDate'].apply(lambda x: 'Before 1.1.2023' if x < 2023 else 'After 1.1.2023')

# Aggregation: Anzahl der markierten Artikel pro Kategorie (EU oder Asien)
df_grouped_eu_asia = df_combined_eu_asia.groupby(['type', 'flag']).size().reset_index(name='count')

# Berechnung der Gesamtzahl der Paper pro Typ (EU/Asien)
df_total_eu_asia = df_combined_eu_asia.groupby('type').size().reset_index(name='total_count')

# Verknüpfe die Gesamtzahlen mit den gruppierten Daten
df_grouped_eu_asia = pd.merge(df_grouped_eu_asia, df_total_eu_asia, on='type')

# Berechne den Prozentsatz der markierten Artikel
df_grouped_eu_asia['percentage'] = (df_grouped_eu_asia['count'] / df_grouped_eu_asia['total_count']) * 100

# Filtere nur die markierten Artikel (Flagged)
df_flagged_eu_asia = df_grouped_eu_asia[df_grouped_eu_asia['flag'] == 'Flagged']

# Aggregation nach der Publikationsdatum-Kategorie (vor und nach 1.1.2023) für markierte Artikel
df_date_grouped_eu_asia = df_combined_eu_asia[df_combined_eu_asia['flag'] == 'Flagged'].groupby(['type', 'DateCategory']).size().reset_index(name='count')

# Berechnung der Gesamtzahl der Paper nach Publikationsdatum für den Prozentsatz
df_total_date_eu_asia = df_combined_eu_asia.groupby(['type', 'DateCategory']).size().reset_index(name='total_count')

# Verknüpfen der Gesamtzahlen mit den gruppierten Daten
df_date_grouped_eu_asia = pd.merge(df_date_grouped_eu_asia, df_total_date_eu_asia, on=['type', 'DateCategory'])

# Berechnung des Prozentsatzes
df_date_grouped_eu_asia['percentage'] = (df_date_grouped_eu_asia['count'] / df_date_grouped_eu_asia['total_count']) * 100

# Filter für die Jahre 2020-2024
df_filtered_years_eu_asia = df_combined_eu_asia[df_combined_eu_asia['PubDate'].isin([2017,2018,2019,2020, 2021, 2022, 2023, 2024])]

# Aggregation der markierten Paper nach Jahr und Typ
df_year_grouped_eu_asia = df_filtered_years_eu_asia[df_filtered_years_eu_asia['flag'] == 'Flagged'].groupby(['type', 'PubDate']).size().reset_index(name='count')

# Berechnung der Gesamtzahl der Paper pro Jahr für den Prozentsatz
df_total_years_eu_asia = df_filtered_years_eu_asia.groupby(['type', 'PubDate']).size().reset_index(name='total_count')

# Verknüpfen der Gesamtzahlen mit den gruppierten Daten
df_year_grouped_eu_asia = pd.merge(df_year_grouped_eu_asia, df_total_years_eu_asia, on=['type', 'PubDate'])

# Berechnung des Prozentsatzes
df_year_grouped_eu_asia['percentage'] = (df_year_grouped_eu_asia['count'] / df_year_grouped_eu_asia['total_count']) * 100
# Perform linear regression to detect trend for Frage 7.2
years_7_2 = df_year_grouped_eu_asia['PubDate'].values
percentages_7_2 = df_year_grouped_eu_asia['percentage'].values
slope_7_2, intercept_7_2, r_value_7_2, p_value_7_2, std_err_7_2 = stats.linregress(years_7_2, percentages_7_2)

# Add trend line to the existing line chart for Frage 7.2
df_year_grouped_eu_asia['trend_7_2'] = intercept_7_2 + slope_7_2 * df_year_grouped_eu_asia['PubDate']

# Display the p-value in the graph title for Frage 7.2
title_with_p_value_7_2 = f"Percentage of Flagged Papers for 2020-2024 (EU vs Asia)\nTrend Line (p-value: {p_value_7_2:.4f})"

#Frage 5
df_kiel_uni_5 = pd.read_csv('Data/Kiel_Uni_arxiv_flag_updated.csv')
df_merged_germany_5 = pd.read_csv('Data/Merged_Germany_datasets.csv')
# Add 'type' column to differentiate CAU and other universities
df_kiel_uni_5['type'] = 'CAU'
df_merged_germany_5['type'] = 'Other Universities'

# Combine the two datasets
df_combined_5 = pd.concat([df_kiel_uni_5, df_merged_germany_5])

# Flag column handling
df_combined_5['flag'] = df_combined_5['flag'].apply(lambda x: 'Flagged' if x == 'Yes' else 'Not Flagged')

# Create date category for before and after 1.1.2023
df_combined_5['DateCategory'] = df_combined_5['PubDate'].apply(lambda x: 'Before 1.1.2023' if x < 2023 else 'After 1.1.2023')

# 1. Bar chart for flagged papers percentage
df_grouped_5 = df_combined_5.groupby(['type', 'flag']).size().reset_index(name='count')
df_total_5 = df_combined_5.groupby('type').size().reset_index(name='total_count')
df_grouped_5 = pd.merge(df_grouped_5, df_total_5, on='type')
df_grouped_5['percentage'] = (df_grouped_5['count'] / df_grouped_5['total_count']) * 100

# 2. Bar chart for comparison before and after 1.1.2023
df_date_grouped_5 = df_combined_5[df_combined_5['flag'] == 'Flagged'].groupby(['type', 'DateCategory']).size().reset_index(name='count')
df_total_date_5 = df_combined_5.groupby(['type', 'DateCategory']).size().reset_index(name='total_count')
df_date_grouped_5 = pd.merge(df_date_grouped_5, df_total_date_5, on=['type', 'DateCategory'])
df_date_grouped_5['percentage'] = (df_date_grouped_5['count'] / df_date_grouped_5['total_count']) * 100

# 3. Line chart for flagged papers for 2020-2024
df_filtered_years_5 = df_combined_5[df_combined_5['PubDate'].isin([2017,2018,2019,2020, 2021, 2022, 2023, 2024])]
df_year_grouped_5 = df_filtered_years_5[df_filtered_years_5['flag'] == 'Flagged'].groupby(['type', 'PubDate']).size().reset_index(name='count')
df_total_years_5 = df_filtered_years_5.groupby(['type', 'PubDate']).size().reset_index(name='total_count')
df_year_grouped_5 = pd.merge(df_year_grouped_5, df_total_years_5, on=['type', 'PubDate'])
df_year_grouped_5['percentage'] = (df_year_grouped_5['count'] / df_year_grouped_5['total_count']) * 100
years = df_year_grouped_5['PubDate'].values
percentages = df_year_grouped_5['percentage'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(years, percentages)
df_year_grouped_5['trend'] = intercept + slope * df_year_grouped_5['PubDate']
title_with_p_value = f"Percentage of Flagged Papers for 2020-2024 (CAU vs Other Universities)\nTrend Line (p-value: {p_value:.4f})"


##### RQ6 Import
RQ_6_df = pd.read_csv('Data/RQ6_faculty_keyword.csv')

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

######## Code for Research Question 1######

# Relative Frequency Calculation for multiple regions
# Function to extract words and their frequencies from the "found_words" column
def extract_words(text):
    pattern = r'(\w+)\s*\((\d+)\)'
    found = re.findall(pattern, text)
    return Counter({word: int(count) for word, count in found})

# Function to calculate relative word frequencies
def calculate_relative_frequencies(df_pre, df_post):
    pre_chatgpt_words = Counter()
    post_chatgpt_words = Counter()

    # Count words for both periods
    for words in df_pre['found_words'].dropna():
        pre_chatgpt_words.update(extract_words(words))

    for words in df_post['found_words'].dropna():
        post_chatgpt_words.update(extract_words(words))

    # Calculate the total number of words for each period
    total_pre_chatgpt_words = sum(pre_chatgpt_words.values())
    total_post_chatgpt_words = sum(post_chatgpt_words.values())

    # Calculate relative frequencies for each word
    pre_chatgpt_relative = {word: count / total_pre_chatgpt_words for word, count in pre_chatgpt_words.items()}
    post_chatgpt_relative = {word: count / total_post_chatgpt_words for word, count in post_chatgpt_words.items()}

    # Ensure both lists have the same words
    all_words = set(pre_chatgpt_relative.keys()).union(set(post_chatgpt_relative.keys()))
    pre_chatgpt_relative = {word: pre_chatgpt_relative.get(word, 0) for word in all_words}
    post_chatgpt_relative = {word: post_chatgpt_relative.get(word, 0) for word in all_words}

    return pre_chatgpt_relative, post_chatgpt_relative, all_words

# Calculate for all regions
df_eu_pre = df_eu[df_eu['PubDate'] <= 2022]
df_eu_post = df_eu[df_eu['PubDate'] >= 2023]
pre_chatgpt_relative_eu, post_chatgpt_relative_eu, all_words_eu = calculate_relative_frequencies(df_eu_pre, df_eu_post)

df_asia_pre = df_asia[df_asia['PubDate'] <= 2022]
df_asia_post = df_asia[df_asia['PubDate'] >= 2023]
pre_chatgpt_relative_asia, post_chatgpt_relative_asia, all_words_asia = calculate_relative_frequencies(df_asia_pre, df_asia_post)

df_universities_pre = df_universities[df_universities['PubDate'] <= 2022]
df_universities_post = df_universities[df_universities['PubDate'] >= 2023]
pre_chatgpt_relative_universities, post_chatgpt_relative_universities, all_words_universities = calculate_relative_frequencies(df_universities_pre, df_universities_post)

df_fachhochschulen_pre = df_fachhochschulen[df_fachhochschulen['PubDate'] <= 2022]
df_fachhochschulen_post = df_fachhochschulen[df_fachhochschulen['PubDate'] >= 2023]
pre_chatgpt_relative_fachhochschulen, post_chatgpt_relative_fachhochschulen, all_words_fachhochschulen = calculate_relative_frequencies(df_fachhochschulen_pre, df_fachhochschulen_post)

# Function to generate bar chart for relative word usage
def generate_relative_frequency_bar(pre_chatgpt_relative, post_chatgpt_relative, region_name):
    # Convert the relative frequency data into a DataFrame
    df = pd.DataFrame({
        'Word': list(pre_chatgpt_relative.keys()),
        'Pre_ChatGPT': list(pre_chatgpt_relative.values()),
        'Post_ChatGPT': list(post_chatgpt_relative.values())
    })

    # Create bar chart using Plotly Express
    fig = px.bar(
        df,
        x='Word',
        y=['Pre_ChatGPT', 'Post_ChatGPT'],
        title=f'Relative Word Usage Before and After ChatGPT ({region_name})',
        barmode='group'
    )

    # Return the figure for use in Dash layout
    return fig

# Generate relative frequency bar charts for all regions

# For EU region
fig_bar_eu = generate_relative_frequency_bar(pre_chatgpt_relative_eu, post_chatgpt_relative_eu, "EU")

# For Asia region
fig_bar_asia = generate_relative_frequency_bar(pre_chatgpt_relative_asia, post_chatgpt_relative_asia, "Asia")

# For Universities
fig_bar_universities = generate_relative_frequency_bar(pre_chatgpt_relative_universities, post_chatgpt_relative_universities, "Universities")

# For Fachhochschulen
fig_bar_fachhochschulen = generate_relative_frequency_bar(pre_chatgpt_relative_fachhochschulen, post_chatgpt_relative_fachhochschulen, "Fachhochschulen")

# Chi-Square Test for multiple regions
# Function to perform the Chi-Square test using relative frequencies
def perform_chi_square_test(pre_chatgpt_relative, post_chatgpt_relative, all_words):
    # Create a contingency table
    contingency_table = pd.DataFrame({
        'Pre_ChatGPT': [pre_chatgpt_relative[word] for word in all_words],
        'Post_ChatGPT': [post_chatgpt_relative[word] for word in all_words]
    })

    # Add a small value to avoid zero frequencies
    contingency_table += 1e-10

    # Perform the Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return chi2, p

# Perform the Chi-Square test for all regions
chi2_eu, p_eu = perform_chi_square_test(pre_chatgpt_relative_eu, post_chatgpt_relative_eu, all_words_eu)
chi2_asia, p_asia = perform_chi_square_test(pre_chatgpt_relative_asia, post_chatgpt_relative_asia, all_words_asia)
chi2_universities, p_universities = perform_chi_square_test(pre_chatgpt_relative_universities, post_chatgpt_relative_universities, all_words_universities)
chi2_fachhochschulen, p_fachhochschulen = perform_chi_square_test(pre_chatgpt_relative_fachhochschulen, post_chatgpt_relative_fachhochschulen, all_words_fachhochschulen)

# Function to generate the Chi-Square pie chart, always showing a visual regardless of p-value
def generate_chi_square_pie(p_value, region_name):
    # If p-value is 1.0, it means "No significant difference"
    if p_value >= 0.05:
        significant_label = "No significant difference"
        significant_value = 1
    else:
        significant_label = "Significant difference"
        significant_value = 1

    # Create pie chart
    fig = go.Figure()

    # Add pie chart trace
    fig.add_trace(go.Pie(
        labels=[significant_label, "Significant difference" if significant_label == "No significant difference" else "No significant difference"],
        values=[significant_value, 0],  # Adjust the second value to ensure the visualization is balanced
        hole=0.4  # Donut chart
    ))

    # Update layout for the chart
    fig.update_layout(
        title_text=f"Chi-Square Test Result for {region_name} (p-value: {p_value:.6f})",
        annotations=[dict(text=significant_label, x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    return fig
# Shapiro-Wilk Test for multiple regions

# Function to perform the Shapiro-Wilk test for normality
def perform_shapiro_test(pre_chatgpt_relative, post_chatgpt_relative):
    pre_chatgpt_values = list(pre_chatgpt_relative.values())
    post_chatgpt_values = list(post_chatgpt_relative.values())

    shapiro_pre_result = None
    shapiro_post_result = None

    # Perform the Shapiro-Wilk test for Pre-ChatGPT data
    if len(pre_chatgpt_values) >= 3:  # Shapiro-Wilk test requires at least 3 values
        stat_pre, p_pre = shapiro(pre_chatgpt_values)
        shapiro_pre_result = (stat_pre, p_pre)

    # Perform the Shapiro-Wilk test for Post-ChatGPT data
    if len(post_chatgpt_values) >= 3:
        stat_post, p_post = shapiro(post_chatgpt_values)
        shapiro_post_result = (stat_post, p_post)

    return shapiro_pre_result, shapiro_post_result

# Perform the Shapiro-Wilk test for all regions
shapiro_pre_eu, shapiro_post_eu = perform_shapiro_test(pre_chatgpt_relative_eu, post_chatgpt_relative_eu)
shapiro_pre_asia, shapiro_post_asia = perform_shapiro_test(pre_chatgpt_relative_asia, post_chatgpt_relative_asia)
shapiro_pre_universities, shapiro_post_universities = perform_shapiro_test(pre_chatgpt_relative_universities, post_chatgpt_relative_universities)
shapiro_pre_fachhochschulen, shapiro_post_fachhochschulen = perform_shapiro_test(pre_chatgpt_relative_fachhochschulen, post_chatgpt_relative_fachhochschulen)

# Function to plot Shapiro-Wilk Test histograms for normality visualization
# Function to generate Plotly histogram for Shapiro-Wilk test
def generate_shapiro_histogram_figure(pre_values, post_values, region_name):
    # Create a figure for Plotly histogram
    fig = go.Figure()

    # Pre-ChatGPT histogram
    fig.add_trace(go.Histogram(x=pre_values, name="Pre-ChatGPT", marker_color='green', opacity=0.75))

    # Post-ChatGPT histogram
    fig.add_trace(go.Histogram(x=post_values, name="Post-ChatGPT", marker_color='blue', opacity=0.75))

    # Update layout
    fig.update_layout(
        title=f'Shapiro-Wilk Test Histogram for {region_name}',
        barmode='overlay',
        xaxis_title='Relative Frequency',
        yaxis_title='Density',
        legend_title="Time Period",
        bargap=0.1,
        bargroupgap=0.2
    )

    # Return the figure
    return fig

# Generate Chi-Square pie charts for all regions
fig_pie_eu = generate_chi_square_pie(p_eu, "EU")
fig_pie_asia = generate_chi_square_pie(p_asia, "Asia")
fig_pie_universities = generate_chi_square_pie(p_universities, "Universities")
fig_pie_fachhochschulen = generate_chi_square_pie(p_fachhochschulen, "Fachhochschulen")

# Generate the Shapiro-Wilk histogram figures for all regions

# EU region
fig_shapiro_eu = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_eu.values()),
    list(post_chatgpt_relative_eu.values()),
    "EU"
)

# Asia region
fig_shapiro_asia = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_asia.values()),
    list(post_chatgpt_relative_asia.values()),
    "Asia"
)

# Universities region
fig_shapiro_universities = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_universities.values()),
    list(post_chatgpt_relative_universities.values()),
    "Universities"
)

# Fachhochschulen region
fig_shapiro_fachhochschulen = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_fachhochschulen.values()),
    list(post_chatgpt_relative_fachhochschulen.values()),
    "Fachhochschulen"
)


############################################ end research question 1 ######



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

        # EU Section
        dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-eu", figure=fig_bar_eu))], className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="chi-square-pie-eu", figure=fig_pie_eu))], className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-eu", figure=fig_shapiro_eu))], className="mb-5"),
        dbc.Row([
            dbc.Col(html.P(
                f"Chi-Square p-value for EU: {p_eu:.6f}. {'Significant difference' if p_eu < 0.05 else 'No significant difference'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Pre-ChatGPT EU: p-value = {shapiro_pre_eu[1]:.6f}. {'Not normal' if shapiro_pre_eu[1] < 0.05 else 'Normal distribution'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Post-ChatGPT EU: p-value = {shapiro_post_eu[1]:.6f}. {'Not normal' if shapiro_post_eu[1] < 0.05 else 'Normal distribution'}.")),
        ], className="mb-5"),

        # Asia Section
        dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-asia", figure=fig_bar_asia))], className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="chi-square-pie-asia", figure=fig_pie_asia))], className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-asia", figure=fig_shapiro_asia))], className="mb-5"),
        dbc.Row([
            dbc.Col(html.P(
                f"Chi-Square p-value for Asia: {p_asia:.6f}. {'Significant difference' if p_asia < 0.05 else 'No significant difference'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Pre-ChatGPT Asia: p-value = {shapiro_pre_asia[1]:.6f}. {'Not normal' if shapiro_pre_asia[1] < 0.05 else 'Normal distribution'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Post-ChatGPT Asia: p-value = {shapiro_post_asia[1]:.6f}. {'Not normal' if shapiro_post_asia[1] < 0.05 else 'Normal distribution'}.")),
        ], className="mb-5"),

        # Universities Section
        dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-universities", figure=fig_bar_universities))], className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="chi-square-pie-universities", figure=fig_pie_universities))], className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-universities", figure=fig_shapiro_universities))],
                className="mb-5"),
        dbc.Row([
            dbc.Col(html.P(
                f"Chi-Square p-value for Universities: {p_universities:.6f}. {'Significant difference' if p_universities < 0.05 else 'No significant difference'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Pre-ChatGPT Universities: p-value = {shapiro_pre_universities[1]:.6f}. {'Not normal' if shapiro_pre_universities[1] < 0.05 else 'Normal distribution'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Post-ChatGPT Universities: p-value = {shapiro_post_universities[1]:.6f}. {'Not normal' if shapiro_post_universities[1] < 0.05 else 'Normal distribution'}.")),
        ], className="mb-5"),

        # Fachhochschulen Section
        dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-fachhochschulen", figure=fig_bar_fachhochschulen))],
                className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="chi-square-pie-fachhochschulen", figure=fig_pie_fachhochschulen))],
                className="mb-5"),
        dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-fachhochschulen", figure=fig_shapiro_fachhochschulen))],
                className="mb-5"),
        dbc.Row([
            dbc.Col(html.P(
                f"Chi-Square p-value for Fachhochschulen: {p_fachhochschulen:.6f}. {'Significant difference' if p_fachhochschulen < 0.05 else 'No significant difference'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Pre-ChatGPT Fachhochschulen: p-value = {shapiro_pre_fachhochschulen[1]:.6f}. {'Not normal' if shapiro_pre_fachhochschulen[1] < 0.05 else 'Normal distribution'}.")),
            dbc.Col(html.P(
                f"Shapiro-Wilk Test for Post-ChatGPT Fachhochschulen: p-value = {shapiro_post_fachhochschulen[1]:.6f}. {'Not normal' if shapiro_post_fachhochschulen[1] < 0.05 else 'Normal distribution'}.")),
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
                "How do the effects of ChatGPT on scientific papers differ between CAU and other German universities? "
                "This includes comparisons between Christian-Albrechts-Universität zu Kiel and other institutions.")),
        ]),
        # Bar chart for flagged papers percentage (Research Question 5)
        dbc.Row([
            dbc.Col(    dcc.Graph(
                id="comparison-graph-5",
                figure=px.bar(df_grouped_5[df_grouped_5['flag'] == 'Flagged'], x='type', y='percentage', color='type',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers: CAU vs Other Universities",
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers'})
            )),
        ], className="mb-5"),

        # Comparison Before and After 1.1.2023 for Research Question 5
        dbc.Row([
            dbc.Col(html.H4("Research Question 5: Comparison of Papers Before and After 1.1.2023 (CAU vs Other Universities)"), width=6),
            dbc.Col(html.P(
                "How has the percentage of flagged papers changed before and after 1.1.2023 between CAU and other German universities? "
                "This graph shows a comparison of flagged papers before and after the introduction of ChatGPT.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="date-comparison-graph-5",
                figure=px.bar(df_date_grouped_5, x='type', y='percentage', color='DateCategory',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers Before and After 1.1.2023 (CAU vs Other Universities)",
                              barmode='group',
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers', 'DateCategory': 'Publication Date Category'})
            )),
        ], className="mb-5"),

        # Line graph for 2020-2024 flagged papers (Research Question 5)
        dbc.Row([
            dbc.Col(html.H4("Research Question 5: Flagged Papers for 2020-2024 (CAU vs Other Universities)"), width=6),
            dbc.Col(html.P(
                "This graph shows the percentage of flagged papers from 2020 to 2024, separated by institution type (CAU and other universities).")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="year-comparison-graph-5-with-trend",
                figure=px.line(df_year_grouped_5, x='PubDate', y='percentage', color='type',
                               color_discrete_sequence=colors_blind_friendly,
                               title=title_with_p_value,  # Add p-value to the title
                               labels={'PubDate': 'Publication Year', 'percentage': 'Percentage of Papers', 'type': 'Institution Type'})
                .add_scatter(x=df_year_grouped_5['PubDate'], y=df_year_grouped_5['trend'], mode='lines', name='Trend Line')
                .update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Set to display full years on the x-axis
  # Set to display whole years on the x-axis
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

            html.Div(id='slider-RQ6', children=
            [dcc.Slider(id='RQ6-year-slider',
                        min=RQ_6_df['Year'].min(),
                        max=RQ_6_df['Year'].max(),
                        value=RQ_6_df['Year'].min(),
                        marks={str(year): str(year) for year in RQ_6_df['Year'].unique()},
                        step=None
                        )], style={'width': '50%', 'display': 'inline-block'}),
            # inline-block : to show slider and dropdown in the same line

            html.Div(id='dropdown-div', children=
            [dcc.Dropdown(id='RQ6-Keyword-dropdown',
                          options=[{'label': i, 'value': i} for i in np.append(['All'], RQ_6_df['Keyword'].unique())],
                          value='All'
                          )], style={'width': '50%', 'display': 'inline-block'}),

            dcc.Graph(id='RQ6-scatter')
        ]),

        # Research Question 7.1: Comparison Between Universities and Fachhochschulen (Percentage of Flagged Papers)
        dbc.Row([
            dbc.Col(html.H4("Research Question 7.1: Global Comparison Between Universities and Fachhochschulen"),
                    width=6),
            dbc.Col(html.P(
                "How do the effects of ChatGPT on scientific papers differ between various universities worldwide? "
                "This includes comparisons between top German universities and Fachhochschulen.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="comparison-graph-7-1",
                figure=px.bar(df_flagged, x='type', y='percentage', color='type',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers between Universities and Fachhochschulen",
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers'})
            )),
        ], className="mb-5"),

        # Comparison Before and After 1.1.2023 for Research Question 7.1
        dbc.Row([
            dbc.Col(html.H4("Research Question 7.1: Comparison of Papers Before and After 1.1.2023"), width=6),
            dbc.Col(html.P(
                "How has the percentage of flagged papers changed before and after 1.1.2023? "
                "This graph shows a comparison of flagged papers before and after the introduction of ChatGPT.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="date-comparison-graph-7-1",
                figure=px.bar(df_date_grouped, x='type', y='percentage', color='DateCategory',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers Before and After 1.1.2023",
                              barmode='group',
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers',
                                      'DateCategory': 'Publication Date Category'})
            )),
        ], className="mb-5"),

        # Line graph for 2020-2024 flagged papers (Research Question 7.1)
        dbc.Row([
            dbc.Col(html.H4("Research Question 7.1: Flagged Papers for 2020-2024"), width=6),
            dbc.Col(html.P(
                "This graph shows the percentage of flagged papers from 2020 to 2024, separated by institution type.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="year-comparison-graph-7-1-with-trend",
                figure=px.line(df_year_grouped, x='PubDate', y='percentage', color='type',
                               color_discrete_sequence=colors_blind_friendly,
                               title=title_with_p_value_7_1,
                               labels={'PubDate': 'Publication Year', 'percentage': 'Percentage of Papers', 'type': 'Institution Type'})
                .add_scatter(x=df_year_grouped['PubDate'], y=df_year_grouped['trend_7_1'], mode='lines', name='Trend Line 7.1')
                .update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Show full years on the x-axis  # Setzt nur ganze Zahlen als X-Achsen-Werte
            )),
        ], className="mb-5"),

        # Research Question 7.2: Comparison Between EU and Asia (Percentage of Flagged Papers)
        dbc.Row([
            dbc.Col(html.H4("Research Question 7.2: Global Comparison Between EU and Asia"), width=6),
            dbc.Col(html.P(
                "How do the effects of ChatGPT on scientific papers differ between European and Asian institutions? "
                "This includes comparisons between top EU universities and Asian universities.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="comparison-graph-7-2",
                figure=px.bar(df_flagged_eu_asia, x='type', y='percentage', color='type',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers between EU and Asia",
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers'})
            )),
        ], className="mb-5"),

        # Comparison Before and After 1.1.2023 for Research Question 7.2
        dbc.Row([
            dbc.Col(html.H4("Research Question 7.2: Comparison of Papers Before and After 1.1.2023 (EU and Asia)"),
                    width=6),
            dbc.Col(html.P(
                "How has the percentage of flagged papers changed before and after 1.1.2023 in EU and Asia? "
                "This graph shows a comparison of flagged papers before and after the introduction of ChatGPT.")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="date-comparison-graph-7-2",
                figure=px.bar(df_date_grouped_eu_asia, x='type', y='percentage', color='DateCategory',
                              color_discrete_sequence=colors_blind_friendly,
                              title="Percentage of Flagged Papers Before and After 1.1.2023 (EU and Asia)",
                              barmode='group',
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers',
                                      'DateCategory': 'Publication Date Category'})
            )),
        ], className="mb-5"),

        # Line graph for 2020-2024 flagged papers (Research Question 7.2)
        dbc.Row([
            dbc.Col(html.H4("Research Question 7.2: Flagged Papers for 2020-2024 (EU and Asia)"), width=6),
            dbc.Col(html.P(
                "This graph shows the percentage of flagged papers from 2020 to 2024, separated by institution type (EU and Asia).")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="year-comparison-graph-7-2-with-trend",
                figure=px.line(df_year_grouped_eu_asia, x='PubDate', y='percentage', color='type',
                               color_discrete_sequence=colors_blind_friendly,
                               title=title_with_p_value_7_2,
                               labels={'PubDate': 'Publication Year', 'percentage': 'Percentage of Papers', 'type': 'Institution Type'})
                .add_scatter(x=df_year_grouped_eu_asia['PubDate'], y=df_year_grouped_eu_asia['trend_7_2'], mode='lines', name='Trend Line 7.2')
                .update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Show full years on the x-axis  # Setzt nur ganze Zahlen als X-Achsen-Werte
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
