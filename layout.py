import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc
from scipy.stats import chi2_contingency, shapiro, norm
from scipy.stats import mannwhitneyu
from ast import literal_eval
from scipy.stats import kstest
import warnings
from collections import Counter
import re
#chatgpt:
# Um  eigene CSV-Datei einzufügen, ersetzen Sie 'your_file.csv' durch den Pfad zu Ihrer CSV-Datei.
# Stellen Sie sicher, dass Ihre CSV-Datei im selben Verzeichnis wie dieses Skript liegt oder geben Sie den vollständigen Pfad an.
# df_bsp = pd.read_csv('your_file.csv')
# df_bsp.head({Anzahl der Zeilen, die angezeigt werden sollen})




colors_blind_friendly = ['#D55E00', '#0072B2', '#F0E442', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#8E44AD',
                         '#F39C12', '#1ABC9C', '#2C3E50', '#C0392B', '#2980B9', '#27AE60']

# DATEN AUS CSV
# test
import plotly.express as px
from scipy import stats

import pandas as pd

# Load all datasets
df_german_uni = pd.read_csv('Data/Frage_7/German_Uni.csv')
df_german_fh = pd.read_csv('Data/Frage_7/German_FH.csv')
df_germany = pd.read_csv('Data/Frage_7/Germany.csv')
df_eu = pd.read_csv('Data/Frage_7/EU.csv')
df_asia = pd.read_csv('Data/Frage_7/Asia.csv')
df_world = pd.read_csv('Data/Frage_7/World.csv')
# Adding 'type' column to differentiate the sources of data
df_german_uni['type'] = 'Universität DE'
df_german_fh['type'] = 'Fachhochschule DE'
df_germany['type'] = 'Germany'
df_eu['type'] = 'EU'
df_asia['type'] = 'Asien'
df_world['type'] = 'World'

# Combine datasets (two-way combinations)

# 1. Combine German University and Fachhochschule
df_combined_Uni_FH = pd.concat([df_german_uni, df_german_fh])

# 2. Combine German University and Germany
df_combined_Uni_Germany = pd.concat([df_german_uni, df_germany])

# 3. Combine German University and EU
df_combined_Uni_EU = pd.concat([df_german_uni, df_eu])

# 4. Combine German University and Asia
df_combined_Uni_Asia = pd.concat([df_german_uni, df_asia])

# 5. Combine German University and World
df_combined_Uni_World = pd.concat([df_german_uni, df_world])

# 6. Combine Fachhochschule and Germany
df_combined_FH_Germany = pd.concat([df_german_fh, df_germany])

# 7. Combine Fachhochschule and EU
df_combined_FH_EU = pd.concat([df_german_fh, df_eu])

# 8. Combine Fachhochschule and Asia
df_combined_FH_Asia = pd.concat([df_german_fh, df_asia])

# 9. Combine Fachhochschule and World
df_combined_FH_World = pd.concat([df_german_fh, df_world])

# 10. Combine Germany and EU
df_combined_Germany_EU = pd.concat([df_germany, df_eu])

# 11. Combine Germany and Asia
df_combined_Germany_Asia = pd.concat([df_germany, df_asia])

# 12. Combine Germany and World
df_combined_Germany_World = pd.concat([df_germany, df_world])

# 13. Combine EU and Asia
df_combined_EU_Asia = pd.concat([df_eu, df_asia])
# 14. Combine EU and World
df_combined_EU_World = pd.concat([df_eu, df_world])

# 15. Combine Asia and World
df_combined_Asia_World = pd.concat([df_asia, df_world])

datasets = {
    'German University': df_german_uni,
    'Fachhochschule': df_german_fh,
    'Germany': df_germany,
    'EU': df_eu,
    'Asia': df_asia,
    'World': df_world
}









df_universities = pd.read_csv('Data/Frage_7/German_Uni.csv')
df_fachhochschulen = pd.read_csv('Data/Frage_7/German_FH.csv')
df_universities['type'] = 'Universität'
df_fachhochschulen['type'] = 'Fachhochschule'


















def prepare_data(df, date_cutoff=2023):
    # Konvertiere 'flag' Spalte in ein besser lesbares Format
    df['flag'] = df['flag'].apply(lambda x: 'Flagged' if x == 'Yes' else 'Not Flagged')

    # Füge eine Spalte für 'DateCategory' hinzu (vor und nach einem bestimmten Datum)
    df['DateCategory'] = df['PubDate'].apply(
        lambda x: f'Before {date_cutoff}' if x < date_cutoff else f'After {date_cutoff}')

    return df


# Schritt 2: Gruppierte Datenberechnung (Flagged Papiere und Gesamtanzahl)
def compute_grouped_data(df):
    # Gruppiere nach Typ (z.B. Universität vs Fachhochschule oder EU vs Asien) und Flag-Status
    df_grouped = df.groupby(['type', 'flag']).size().reset_index(name='count')
    df_total = df.groupby('type').size().reset_index(name='total_count')

    # Prozentsätze berechnen
    df_grouped = pd.merge(df_grouped, df_total, on='type')
    df_grouped['percentage'] = (df_grouped['count'] / df_grouped['total_count']) * 100

    return df_grouped


# Schritt 3: Jährliche Trends (Flagged Papiere über bestimmte Jahre hinweg)
def compute_yearly_trend(df, years_range):
    # Filter the data for the specified years and remove rows where PubDate is invalid (PubDate == 0)
    df_filtered_years = df[(df['PubDate'].isin(years_range)) & (df['PubDate'] != 0)]

    # Group by year and type for flagged papers
    df_year_grouped = df_filtered_years[df_filtered_years['flag'] == 'Flagged'].groupby(
        ['type', 'PubDate']).size().reset_index(name='count')
    df_total_years = df_filtered_years.groupby(['type', 'PubDate']).size().reset_index(name='total_count')

    # Merge the two dataframes
    df_year_grouped = pd.merge(df_year_grouped, df_total_years, on=['type', 'PubDate'])

    # Calculate percentages
    df_year_grouped['percentage'] = (df_year_grouped['count'] / df_year_grouped['total_count']) * 100

    # Check if there's enough data to perform regression
    if df_year_grouped.shape[0] < 2:
        return df_year_grouped, None  # Not enough data for regression

    # Perform linear regression
    years = df_year_grouped['PubDate'].values
    percentages = df_year_grouped['percentage'].values

    if len(years) == 0 or len(percentages) == 0:
        return df_year_grouped, None  # Avoid running regression on empty arrays

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, percentages)

    # Add trend line
    df_year_grouped['trend'] = intercept + slope * df_year_grouped['PubDate']

    return df_year_grouped, p_value



# Schritt 4: Visualisierungen
def visualize_task(df_grouped, df_year_grouped, p_value, task_title):
    # Bar-Diagramm für markierte Papiere
    fig_bar = px.bar(df_grouped[df_grouped['flag'] == 'Flagged'], x='type', y='percentage', color='type',
                     title=f"Percentage of Flagged Papers ({task_title})",
                     labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers'})

    # Liniendiagramm für markierte Papiere über die Jahre mit Trendlinie
    if p_value is not None:
        title_with_p_value = f"Percentage of Flagged Papers ({task_title}) from {df_year_grouped['PubDate'].min()} to {df_year_grouped['PubDate'].max()} (Trend Line, p-value: {p_value:.4f})"
    else:
        title_with_p_value = f"Percentage of Flagged Papers ({task_title}) from {df_year_grouped['PubDate'].min()} to {df_year_grouped['PubDate'].max()} (No valid p-value)"

    fig_line = px.line(df_year_grouped, x='PubDate', y='percentage', color='type',
                       title=title_with_p_value,
                       labels={'PubDate': 'Publication Year', 'percentage': 'Percentage of Papers',
                               'type': 'Institution Type'})

    # Trendlinie hinzufügen
    fig_line.add_scatter(x=df_year_grouped['PubDate'], y=df_year_grouped['trend'], mode='lines', name='Trend Line')

    return fig_bar, fig_line



# Schritt 5: Universeller Workflow für jede Aufgabe (z.B. 7.1, 7.2)
def task_workflow(df, years_range, task_title):
    df = prepare_data(df)
    df_grouped = compute_grouped_data(df)
    df_year_grouped, p_value = compute_yearly_trend(df, years_range)
    fig_bar, fig_line = visualize_task(df_grouped, df_year_grouped, p_value, task_title)

    return fig_bar, fig_line


# Assuming df_combined is the merged dataframe of universities and Fachhochschulen
#fig_bar_7_1, fig_line_7_1 = task_workflow(df_combined_Uni_FH, [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
#                                          "Universities vs Fachhochschulen")
#fig_bar_7_2, fig_line_7_2 = task_workflow(df_combined_Asia_World, [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
#                                          "EU vs Asia")

# Frage 5
df_kiel_uni_5 = pd.read_csv('Data/Kiel_Uni_arxiv_flag_updated.csv')
df_merged_germany_5 = pd.read_csv('Data/Frage_7/Germany.csv')
# Add 'type' column to differentiate CAU and other universities
df_kiel_uni_5['type'] = 'CAU'
df_merged_germany_5['type'] = 'Other Universities'

# Combine the two datasets
df_combined_5 = pd.concat([df_kiel_uni_5, df_merged_germany_5])

# Flag column handling
df_combined_5['flag'] = df_combined_5['flag'].apply(lambda x: 'Flagged' if x == 'Yes' else 'Not Flagged')

# Create date category for before and after 1.1.2023
df_combined_5['DateCategory'] = df_combined_5['PubDate'].apply(
    lambda x: 'Before 1.1.2023' if x < 2023 else 'After 1.1.2023')

# 1. Bar chart for flagged papers percentage
df_grouped_5 = df_combined_5.groupby(['type', 'flag']).size().reset_index(name='count')
df_total_5 = df_combined_5.groupby('type').size().reset_index(name='total_count')
df_grouped_5 = pd.merge(df_grouped_5, df_total_5, on='type')
df_grouped_5['percentage'] = (df_grouped_5['count'] / df_grouped_5['total_count']) * 100

# 2. Bar chart for comparison before and after 1.1.2023
df_date_grouped_5 = df_combined_5[df_combined_5['flag'] == 'Flagged'].groupby(
    ['type', 'DateCategory']).size().reset_index(name='count')
df_total_date_5 = df_combined_5.groupby(['type', 'DateCategory']).size().reset_index(name='total_count')
df_date_grouped_5 = pd.merge(df_date_grouped_5, df_total_date_5, on=['type', 'DateCategory'])
df_date_grouped_5['percentage'] = (df_date_grouped_5['count'] / df_date_grouped_5['total_count']) * 100

# 3. Line chart for flagged papers for 2020-2024
df_filtered_years_5 = df_combined_5[df_combined_5['PubDate'].isin([2012,2013,2014,2015,2016,2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])]
df_year_grouped_5 = df_filtered_years_5[df_filtered_years_5['flag'] == 'Flagged'].groupby(
    ['type', 'PubDate']).size().reset_index(name='count')
df_total_years_5 = df_filtered_years_5.groupby(['type', 'PubDate']).size().reset_index(name='total_count')
df_year_grouped_5 = pd.merge(df_year_grouped_5, df_total_years_5, on=['type', 'PubDate'])
df_year_grouped_5['percentage'] = (df_year_grouped_5['count'] / df_year_grouped_5['total_count']) * 100
years = df_year_grouped_5['PubDate'].values
percentages = df_year_grouped_5['percentage'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(years, percentages)
df_year_grouped_5['trend'] = intercept + slope * df_year_grouped_5['PubDate']
title_with_p_value = f"Percentage of Flagged Papers for 2017-2024 (CAU vs Other Universities)\nTrend Line (p-value: {p_value:.4f})"

# Research question 4
df_RQ4_comparison = pd.read_csv('Data/RQ4+6/RQ4_comparison.csv')
RQ_4_df = pd.read_csv('Data/RQ4+6/RQ4_PDF_Abstract.csv')

##### RQ6 Import
RQ_6_df = pd.read_csv('Data/RQ4+6/RQ6_faculty_keyword.csv')

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

# List of color codes for color-blind friendly palette
colors_blind_friendly_extended = ['#D55E00', '#0072B2', '#F0E442', '#009E73', '#E69F00', '#56B4E9', '#CC79A7',
                                  '#8E44AD', '#F39C12', '#1ABC9C', '#2C3E50', '#C0392B', '#2980B9', '#27AE60']

# Use the blue and orange tones specifically for the visualizations

color_orange = colors_blind_friendly[0]  # '#D55E00' (Orange)
color_blue = colors_blind_friendly[1]  # '#0072B2' (Blue)
color_yellow = colors_blind_friendly[2]  # yellow
color_green = colors_blind_friendly[3] # green


######## Code for Research Question 1#####################################################

df_world_q1 = pd.read_csv("Data/Merged_World_datasets.csv")

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
pre_chatgpt_relative_asia, post_chatgpt_relative_asia, all_words_asia = calculate_relative_frequencies(df_asia_pre,
                                                                                                       df_asia_post)

# For World (Newly Added World Data)
df_world_q1_pre = df_world_q1[df_world_q1['PubDate'] <= 2022]
df_world_q1_post = df_world_q1[df_world_q1['PubDate'] >= 2023]
pre_chatgpt_relative_world_q1, post_chatgpt_relative_world_q1, all_words_world_q1 = calculate_relative_frequencies(df_world_q1_pre, df_world_q1_post)


df_universities_pre = df_universities[df_universities['PubDate'] <= 2022]
df_universities_post = df_universities[df_universities['PubDate'] >= 2023]
pre_chatgpt_relative_universities, post_chatgpt_relative_universities, all_words_universities = calculate_relative_frequencies(
    df_universities_pre, df_universities_post)

df_fachhochschulen_pre = df_fachhochschulen[df_fachhochschulen['PubDate'] <= 2022]
df_fachhochschulen_post = df_fachhochschulen[df_fachhochschulen['PubDate'] >= 2023]
pre_chatgpt_relative_fachhochschulen, post_chatgpt_relative_fachhochschulen, all_words_fachhochschulen = calculate_relative_frequencies(
    df_fachhochschulen_pre, df_fachhochschulen_post)


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
        title=f'Relative Word Usage Before and After ChatGPT for {region_name}',
        barmode='group',
        color_discrete_map = {
            'Pre_ChatGPT': color_blue,
            'Post_ChatGPT': color_orange
        }
    )

    # Return the figure for use in Dash layout
    return fig

# Generate relative frequency bar charts for all regions

# For EU region
fig_bar_eu = generate_relative_frequency_bar(pre_chatgpt_relative_eu, post_chatgpt_relative_eu, "EU universities")

# For Asia region
fig_bar_asia = generate_relative_frequency_bar(pre_chatgpt_relative_asia, post_chatgpt_relative_asia, "Asia universities")

# For World (Newly Added World Data)
fig_bar_world_q1 = generate_relative_frequency_bar(pre_chatgpt_relative_world_q1, post_chatgpt_relative_world_q1, "World universities")

# For Universities
fig_bar_universities = generate_relative_frequency_bar(pre_chatgpt_relative_universities, post_chatgpt_relative_universities, "German universities")

# For Fachhochschulen
fig_bar_fachhochschulen = generate_relative_frequency_bar(pre_chatgpt_relative_fachhochschulen, post_chatgpt_relative_fachhochschulen, "German universities of applied sciences")


# Chi-Square Test function
# Function to perform Chi-Square test for word groups


def perform_chi_square_test(pre_chatgpt_relative, post_chatgpt_relative, all_words):
    # Prepare lists to store observed (post) and expected (pre) relative frequencies
    observed_values = [post_chatgpt_relative[word] for word in all_words]
    expected_values = [pre_chatgpt_relative[word] for word in all_words]

    # Create a contingency table with relative frequencies
    contingency_table = pd.DataFrame({
        'Expected (Pre_ChatGPT)': expected_values,
        'Observed (Post_ChatGPT)': observed_values
    })

    # Ensure no zero counts by adding a small value (optional)
    contingency_table += 1e-10

    # Perform the Chi-Square test (returning chi2 and p-value)
    chi2, p, dof, expected = chi2_contingency(contingency_table.T)

    # Calculate the residuals for the heatmap (Observed - Expected)
    residuals = np.array(observed_values) - np.array(expected[1])

    return chi2, p, dof, residuals


chi2_universities, p_universities, dof_universities, residuals_universities = perform_chi_square_test(pre_chatgpt_relative_universities,
                                                                                                       post_chatgpt_relative_universities,
                                                                                                       all_words_universities)

chi2_fachhochschulen, p_fachhochschulen, dof_fachhochschulen, residuals_fachhochschulen = perform_chi_square_test(pre_chatgpt_relative_fachhochschulen,
                                                                                                                 post_chatgpt_relative_fachhochschulen,
                                                                                                                 all_words_fachhochschulen)
def generate_chi_square_heatmap(residuals, words, region_name):
    # Ensure that words are passed as a list, not a set
    words = list(words)  # Convert set to list if necessary

    # Create a 2D array for the heatmap
    heatmap_data = [residuals]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,  # Heatmap data
        x=words,  # X-axis: words/categories
        y=[region_name],  # Y-axis (only one row for the region name)
        colorscale='Blues',  # Color scale
        colorbar=dict(title="Residuals")  # Colorbar with title
    ))

    # Add annotation for the explanation under the title
    fig.add_annotation(
        text="Lighter bars indicate smaller deviations, darker bars indicate larger deviations.",
        xref="paper", yref="paper",
        x=0.5, y=1.15,  # Position the text directly under the title
        showarrow=False,
        font=dict(size=12),
        xanchor='center'
    )

    # Update layout for the heatmap
    fig.update_layout(
        title_text=f"Chi-Square Residual Heatmap for {region_name}",
        xaxis_title="Words",  # Or any other appropriate title for x-axis
        yaxis=dict(
            title="Region Name",  # Label for the y-axis
            title_standoff=25,  # Adjust the space between the y-axis and the label
            tickangle=-90  # Rotate the y-axis label so it reads from bottom to top
        ),
        xaxis_tickangle=-45  # Rotate word labels on the x-axis
    )
    return fig

# Perform the Chi-Square test for different regions and generate heatmaps

# Universities
chi2_universities, p_universities, dof_universities, residuals_universities = perform_chi_square_test(pre_chatgpt_relative_universities, post_chatgpt_relative_universities, all_words_universities)
fig_heatmap_universities = generate_chi_square_heatmap(residuals_universities, all_words_universities, "German universities")

# EU
chi2_eu, p_eu, dof_eu, residuals_eu = perform_chi_square_test(pre_chatgpt_relative_eu, post_chatgpt_relative_eu, all_words_eu)
fig_heatmap_eu = generate_chi_square_heatmap(residuals_eu, all_words_eu, "EU universities")

# Asia
chi2_asia, p_asia, dof_asia, residuals_asia = perform_chi_square_test(pre_chatgpt_relative_asia, post_chatgpt_relative_asia, all_words_asia)
fig_heatmap_asia = generate_chi_square_heatmap(residuals_asia, all_words_asia, "Asia universities")


# Now also World Q1
chi2_world_q1, p_world_q1, dof_world_q1, residuals_world_q1 = perform_chi_square_test(pre_chatgpt_relative_world_q1, post_chatgpt_relative_world_q1, all_words_world_q1)
fig_heatmap_world_q1 = generate_chi_square_heatmap(residuals_world_q1, all_words_world_q1, "World universities")

# Fachhochschulen
chi2_fachhochschulen, p_fachhochschulen, dof_fachhochschulen, residuals_fachhochschulen = perform_chi_square_test(pre_chatgpt_relative_fachhochschulen, post_chatgpt_relative_fachhochschulen, all_words_fachhochschulen)
fig_heatmap_fachhochschulen = generate_chi_square_heatmap(residuals_fachhochschulen, all_words_fachhochschulen, "German universities of applied sciences")

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
# Now include World Q1
shapiro_pre_world_q1, shapiro_post_world_q1 = perform_shapiro_test(pre_chatgpt_relative_world_q1, post_chatgpt_relative_world_q1)


# Function to plot Shapiro-Wilk Test histograms for normality visualization
# Function to generate histogram for Shapiro-Wilk test with a normal distribution line
def generate_shapiro_histogram_figure(pre_values, post_values, region_name):
    # Create a figure for Plotly histogram
    fig = go.Figure()

    # Pre-ChatGPT histogram
    fig.add_trace(go.Histogram(x=pre_values, name="Pre-ChatGPT", marker_color=color_blue, opacity=0.75))

    # Post-ChatGPT histogram
    fig.add_trace(go.Histogram(x=post_values, name="Post-ChatGPT", marker_color=color_orange, opacity=0.75))

    # Generate a normal distribution line for Pre-ChatGPT
    x_pre = np.linspace(min(pre_values), max(pre_values), 100)
    mean_pre = np.mean(pre_values)
    std_pre = np.std(pre_values)
    y_pre = (1 / (std_pre * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pre - mean_pre) / std_pre) ** 2)
    fig.add_trace(go.Scatter(x=x_pre, y=y_pre, mode='lines', name='Theoretical normal Distribution (Pre-ChatGPT)', line=dict(color='blue', width=2)))

    # Generate a normal distribution line for Post-ChatGPT
    x_post = np.linspace(min(post_values), max(post_values), 100)
    mean_post = np.mean(post_values)
    std_post = np.std(post_values)
    y_post = (1 / (std_post * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_post - mean_post) / std_post) ** 2)
    fig.add_trace(go.Scatter(x=x_post, y=y_post, mode='lines', name='Theoretical normal Distribution (Post-ChatGPT)', line=dict(color='orange', width=2)))

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

    return fig
# Generate the Shapiro-Wilk histogram figures for all regions

# EU region
fig_shapiro_eu = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_eu.values()),
    list(post_chatgpt_relative_eu.values()),
    "EU universities"
)

# Asia region
fig_shapiro_asia = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_asia.values()),
    list(post_chatgpt_relative_asia.values()),
    "Asia universities"
)

# Universities region
fig_shapiro_universities = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_universities.values()),
    list(post_chatgpt_relative_universities.values()),
    "German universities"
)

# Fachhochschulen region
fig_shapiro_fachhochschulen = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_fachhochschulen.values()),
    list(post_chatgpt_relative_fachhochschulen.values()),
    "German universities of applied sciences"
)

# World Q1 region
fig_shapiro_world_q1 = generate_shapiro_histogram_figure(
    list(pre_chatgpt_relative_world_q1.values()),
    list(post_chatgpt_relative_world_q1.values()),
    "World universities"
)

def clean_numeric_values(values):
    # Filter out any non-numeric values or NaNs
    return [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]

# Mann-Whitney U-Test function
def perform_mann_whitney_test(pre_values, post_values):
    # Clean the data to ensure only numeric values are passed to the test
    pre_values_clean = clean_numeric_values(pre_values)
    post_values_clean = clean_numeric_values(post_values)

    # Perform the Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(pre_values_clean, post_values_clean)
    return u_stat, p_value


# Violin plot for visualizing distributions
def generate_violin_plot(pre_values, post_values, region_name):
    # Create a DataFrame for Plotly
    data = pd.DataFrame({
        'Values': pre_values + post_values,
        'Group': ['Pre-ChatGPT'] * len(pre_values) + ['Post-ChatGPT'] * len(post_values)
    })

    # Create the violin plot with custom colors
    fig = px.violin(data, x='Group', y='Values', box=True, points='all', title=f"Mann-Whitney U Test for {region_name}",
                    color='Group', color_discrete_map={'Pre-ChatGPT': color_blue, 'Post-ChatGPT': color_orange})

    # Customize layout

    return fig

# Perform the Mann-Whitney U test for all regions

# For EU
u_stat_eu, p_value_eu = perform_mann_whitney_test(list(pre_chatgpt_relative_eu.values()), list(post_chatgpt_relative_eu.values()))
fig_violin_eu = generate_violin_plot(list(pre_chatgpt_relative_eu.values()), list(post_chatgpt_relative_eu.values()), "EU universities")

# For Asia
u_stat_asia, p_value_asia = perform_mann_whitney_test(list(pre_chatgpt_relative_asia.values()), list(post_chatgpt_relative_asia.values()))
fig_violin_asia = generate_violin_plot(list(pre_chatgpt_relative_asia.values()), list(post_chatgpt_relative_asia.values()), "Asia universities")

# For Universities
u_stat_universities, p_value_universities = perform_mann_whitney_test(list(pre_chatgpt_relative_universities.values()), list(post_chatgpt_relative_universities.values()))
fig_violin_universities = generate_violin_plot(list(pre_chatgpt_relative_universities.values()), list(post_chatgpt_relative_universities.values()), "German universities")

# For Fachhochschulen
u_stat_fachhochschulen, p_value_fachhochschulen = perform_mann_whitney_test(list(pre_chatgpt_relative_fachhochschulen.values()), list(post_chatgpt_relative_fachhochschulen.values()))
fig_violin_fachhochschulen = generate_violin_plot(list(pre_chatgpt_relative_fachhochschulen.values()), list(post_chatgpt_relative_fachhochschulen.values()), "German universities of applied sciences")

u_stat_world_q1, p_value_world_q1 = perform_mann_whitney_test(
    list(pre_chatgpt_relative_world_q1.values()),
    list(post_chatgpt_relative_world_q1.values())
)

fig_violin_world_q1 = generate_violin_plot(
    list(pre_chatgpt_relative_world_q1.values()),
    list(post_chatgpt_relative_world_q1.values()),
    "World universities"
)

############################################ End research question 1 ##################################################


######## Code for Research Question 2 ###########################################################################

# Load the CSV files for each region
df_eu_question = pd.read_csv('Data/Merged_EU_datasets_questionwords.csv')
df_asia_question = pd.read_csv('Data/Merged_Asia_datasets_questionwords.csv')
df_fachhochschule_question = pd.read_csv('Data/Merged_FH_datasets_questionwords.csv')
df_world_question = pd.read_csv('Data/Merged_World_datasets_questionwords.csv')
df_germany_question= pd.read_csv('Data/Merged_Germany_datasets_questionwords.csv')

# Function to extract question words and their counts from the 'Question_Words' column
def extract_question_words(question_words_column):
    question_words_count = []
    for entry in question_words_column:
        if isinstance(entry, str):  # Check if the entry is a valid string
            words = re.findall(r'(\w+)\s\((\d+)\)', entry)
            word_count = {word: int(count) for word, count in words}
            question_words_count.append(word_count)
        else:
            question_words_count.append({})
    return question_words_count

# Function to aggregate question words across the dataframe
def aggregate_question_words(df):
    total_words = Counter()
    for index, row in df.iterrows():
        total_words.update(row['Question_Words_Count'])
    return total_words

# Function to calculate relative frequencies of question words
def calculate_relative_frequencies(word_counts, num_abstracts):
    return {word: count / num_abstracts for word, count in word_counts.items()}

# Function to calculate relative frequencies of question marks
def calculate_relative_marks(df, num_abstracts):
    return df['Question_Mark_Count'].sum() / num_abstracts
# Process each region separately

# Function to generate bar chart for question words with the specified colors
def generate_question_word_bar_chart(before_values, after_values, words,region_name_q):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=words, y=before_values, name='Pre-ChatGPT', marker_color=color_blue))
    fig.add_trace(go.Bar(x=words, y=after_values, name='Post-ChatGPT', marker_color=color_orange))

    # Update layout
    fig.update_layout(
        title=f"Comparison of Relative Frequencies of Question Words (Pre-ChatGPT vs Post-ChatGPT) <br> for {region_name_q}",
        xaxis_title="Question Words",
        yaxis_title="Relative Frequency",
        barmode='group',
        xaxis_tickangle=-45
    )

    return fig

# Bar chart for question marks with Pre-ChatGPT and Post-ChatGPT labels, using specified colors
def generate_question_mark_chart(relative_marks_before, relative_marks_after, region_name_q):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Pre-ChatGPT', 'Post-ChatGPT'], y=[relative_marks_before, relative_marks_after],
                         marker_color=[color_blue, color_orange]))

    # Update layout
    fig.update_layout(
        title= f"Comparison of Relative Frequencies of Question Marks (Pre-ChatGPT vs Post-ChatGPT) <br> for {region_name_q}",
        xaxis_title="Time Period",
        yaxis_title="Relative Frequency"
    )

    return fig

# Process each region separately using the correct DataFrame
def process_region_data(df, region_name):
    # Check if 'Question_Words' column exists
    if 'Question_Words' not in df.columns:
        raise KeyError(f"Column 'Question_Words' not found in {region_name} data")

    # Extract question words
    df['Question_Words_Count'] = extract_question_words(df['Question_Words'])

    # Split data into before and after 2023
    df_before_2023 = df[df['PubDate'] <= 2022]
    df_after_2023 = df[df['PubDate'] >= 2023]

    # Number of abstracts
    n_abstracts_before_2023 = len(df_before_2023)
    n_abstracts_after_2023 = len(df_after_2023)

    # Aggregate question words
    words_before_2023 = aggregate_question_words(df_before_2023)
    words_after_2023 = aggregate_question_words(df_after_2023)

    # Filter out excluded words
    excluded_words = {'much', 'many', 'far', 'long'}
    words_before_2023_filtered = {k: v for k, v in words_before_2023.items() if k not in excluded_words}
    words_after_2023_filtered = {k: v for k, v in words_after_2023.items() if k not in excluded_words}

    # Calculate relative frequencies
    relative_words_before_2023 = calculate_relative_frequencies(words_before_2023_filtered, n_abstracts_before_2023)
    relative_words_after_2023 = calculate_relative_frequencies(words_after_2023_filtered, n_abstracts_after_2023)

    # Calculate question mark frequencies
    relative_marks_before_2023 = calculate_relative_marks(df_before_2023, n_abstracts_before_2023)
    relative_marks_after_2023 = calculate_relative_marks(df_after_2023, n_abstracts_after_2023)

    # Prepare data for visualization
    all_words = sorted(set(relative_words_before_2023.keys()).union(set(relative_words_after_2023.keys())))
    before_values = [relative_words_before_2023.get(word, 0) for word in all_words]
    after_values = [relative_words_after_2023.get(word, 0) for word in all_words]

    # Generate figures with region_name_q
    fig_question_words = generate_question_word_bar_chart(before_values, after_values, all_words, region_name)
    fig_question_marks = generate_question_mark_chart(relative_marks_before_2023, relative_marks_after_2023, region_name)

    return fig_question_words, fig_question_marks


# Process the regions using the specific variables for each
fig_words_eu, fig_marks_eu = process_region_data(df_eu_question, "EU")
fig_words_asia, fig_marks_asia = process_region_data(df_asia_question, "Asia")
fig_words_fachhochschule, fig_marks_fachhochschule = process_region_data(df_fachhochschule_question, "German universities of applied Science")
fig_words_world, fig_marks_world = process_region_data(df_world_question, "World")
fig_words_germany, fig_marks_germany = process_region_data(df_germany_question, "German universities")

# Function to perform the Chi-Square test with relative frequencies
def perform_chi_square_test_question_words(before_values, after_values, all_words, region_name):
    # Print the relative frequencies for checking
    #print(f"Before ChatGPT Frequencies ({region_name}): {before_values}")
    #print(f"After ChatGPT Frequencies ({region_name}): {after_values}")

    # Create a contingency table with relative frequencies
    contingency_table = pd.DataFrame({
        'Pre_ChatGPT': before_values,
        'Post_ChatGPT': after_values
    })

    # Check for zero or extremely small values in the contingency table
    #print(f"Contingency Table ({region_name}):\n{contingency_table}")

    # Add a small value to avoid zero frequencies (if necessary)
    contingency_table += 1e-10

    # Perform the Chi-Square test (returning chi2 and p-value)
    chi2, p, dof, _ = chi2_contingency(contingency_table.T)

    # Print Chi-Square statistics for debugging
    #print(f"Chi2 = {chi2}, p-value = {p}, dof = {dof} ({region_name})")

    # Create heatmap of residuals (observed - expected)
    residuals = np.array(after_values) - np.array(before_values)

    # Generate heatmap using residuals

    fig = go.Figure(data=go.Heatmap(
        z=[residuals],  # Heatmap data
        x=all_words,  # X-axis: words/categories
        y=[region_name],  # Y-axis (only one row for the region name)
        colorscale='Blues',  # Default blue scale
        colorbar=dict(title="Residuals")  # Colorbar with title, here is
    ))

    # Add annotation for the explanation under the title
    fig.add_annotation(
        text="Lighter bars indicate smaller deviations, darker bars indicate larger deviations.",
        xref="paper", yref="paper",
        x=0.5, y=1.15,  # Position the text directly under the title
        showarrow=False,
        font=dict(size=12),
        xanchor='center'
    )


    # Update layout for the heatmap
    fig.update_layout(
        title_text=f"Chi-Square Residual Heatmap for {region_name}",
        xaxis_title="Question Words",
        yaxis=dict(
            title="Region Name",  # Label for the y-axis
            title_standoff=25,  # Space between axis and label
            tickangle=-90  # Rotate the label from bottom to top
        ),
        xaxis_tickangle=-45  # Rotate word labels on x-axis
    )


    return chi2, p, dof, fig


# Process the regions using the specific variables for each
def process_region_with_chi_square(df, region_name):
    # Extract question words
    df['Question_Words_Count'] = extract_question_words(df['Question_Words'])

    # Split data into before and after 2023
    df_before_2023 = df[df['PubDate'] <= 2022]
    df_after_2023 = df[df['PubDate'] >= 2023]

    # Number of abstracts
    n_abstracts_before_2023 = len(df_before_2023)
    n_abstracts_after_2023 = len(df_after_2023)

    # Aggregate question words
    words_before_2023 = aggregate_question_words(df_before_2023)
    words_after_2023 = aggregate_question_words(df_after_2023)

    # Filter out excluded words
    excluded_words = {'much', 'many', 'far', 'long'}
    words_before_2023_filtered = {k: v for k, v in words_before_2023.items() if k not in excluded_words}
    words_after_2023_filtered = {k: v for k, v in words_after_2023.items() if k not in excluded_words}

    # Calculate relative frequencies
    relative_words_before_2023 = calculate_relative_frequencies(words_before_2023_filtered, n_abstracts_before_2023)
    relative_words_after_2023 = calculate_relative_frequencies(words_after_2023_filtered, n_abstracts_after_2023)

    # Prepare data for Chi-Square test
    all_words = sorted(set(relative_words_before_2023.keys()).union(set(relative_words_after_2023.keys())))
    before_values = [relative_words_before_2023.get(word, 0) for word in all_words]
    after_values = [relative_words_after_2023.get(word, 0) for word in all_words]

    # Perform Chi-Square test and generate heatmap
    chi2, p, dof, fig_heatmap = perform_chi_square_test_question_words(before_values, after_values, all_words,
                                                                       region_name)

    return chi2, p, dof, fig_heatmap


# Process the regions and perform Chi-Square tests for Research Question 2
chi2_q2_eu, p_q2_eu, dof_q2_eu, fig_heatmap_q2_eu = process_region_with_chi_square(df_eu_question, "EU")
chi2_q2_asia, p_q2_asia, dof_q2_asia, fig_heatmap_q2_asia = process_region_with_chi_square(df_asia_question, "Asia")
chi2_q2_fachhochschule, p_q2_fachhochschule, dof_q2_fachhochschule, fig_heatmap_q2_fachhochschule = process_region_with_chi_square(
    df_fachhochschule_question, "German universities of applied Science")
chi2_q2_world, p_q2_world, dof_q2_world, fig_heatmap_q2_world = process_region_with_chi_square(df_world_question, "World")
chi2_q2_germany, p_q2_germany, dof_q2_germany, fig_heatmap_q2_germany = process_region_with_chi_square(df_germany_question, "Germany universities")

# Function to perform the Shapiro-Wilk test for normality
def perform_shapiro_test(pre_values, post_values):
    shapiro_pre_result = None
    shapiro_post_result = None

    # Perform the Shapiro-Wilk test for Pre-ChatGPT data
    if len(pre_values) >= 3:  # Shapiro-Wilk test requires at least 3 values
        stat_pre, p_pre = shapiro(pre_values)
        shapiro_pre_result = (stat_pre, p_pre)

    # Perform the Shapiro-Wilk test for Post-ChatGPT data
    if len(post_values) >= 3:
        stat_post, p_post = shapiro(post_values)
        shapiro_post_result = (stat_post, p_post)

    return shapiro_pre_result, shapiro_post_result

# Function to generate histogram for Shapiro-Wilk test with a normal distribution line
def generate_shapiro_histogram_figure(pre_values, post_values, region_name):
    # Create a figure for Plotly histogram
    fig = go.Figure()

    # Pre-ChatGPT histogram
    fig.add_trace(go.Histogram(x=pre_values, name="Pre-ChatGPT", marker_color=color_blue, opacity=0.75))

    # Post-ChatGPT histogram
    fig.add_trace(go.Histogram(x=post_values, name="Post-ChatGPT", marker_color=color_orange, opacity=0.75))

    # Generate a normal distribution line for comparison
    x = np.linspace(min(min(pre_values), min(post_values)), max(max(pre_values), max(post_values)), 100)
    mean_pre = np.mean(pre_values)
    std_pre = np.std(pre_values)
    y_pre = (1 / (std_pre * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_pre) / std_pre) ** 2)

    mean_post = np.mean(post_values)
    std_post = np.std(post_values)
    y_post = (1 / (std_post * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_post) / std_post) ** 2)

    # Add normal distribution lines to the plot
    fig.add_trace(go.Scatter(x=x, y=y_pre, mode='lines', name='Theoretical normal Distribution (Pre-ChatGPT)', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_post, mode='lines', name='Theoretical normal Distribution (Post-ChatGPT)', line=dict(color='orange', width=2)))

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

    return fig

# Process the regions using the specific variables for each
def process_region_with_shapiro_test(df, region_name):
    # Extract question words
    df['Question_Words_Count'] = extract_question_words(df['Question_Words'])

    # Split data into before and after 2023
    df_before_2023 = df[df['PubDate'] <= 2022]
    df_after_2023 = df[df['PubDate'] >= 2023]

    # Aggregate question words
    words_before_2023 = aggregate_question_words(df_before_2023)
    words_after_2023 = aggregate_question_words(df_after_2023)

    # Filter out excluded words
    excluded_words = {'much', 'many', 'far', 'long'}
    words_before_2023_filtered = {k: v for k, v in words_before_2023.items() if k not in excluded_words}
    words_after_2023_filtered = {k: v for k, v in words_after_2023.items() if k not in excluded_words}

    # Calculate relative frequencies
    relative_words_before_2023 = list(calculate_relative_frequencies(words_before_2023_filtered, len(df_before_2023)).values())
    relative_words_after_2023 = list(calculate_relative_frequencies(words_after_2023_filtered, len(df_after_2023)).values())

    # Perform Shapiro-Wilk test
    shapiro_pre_result, shapiro_post_result = perform_shapiro_test(relative_words_before_2023, relative_words_after_2023)

    # Generate histogram with normal distribution
    fig_shapiro_histogram = generate_shapiro_histogram_figure(relative_words_before_2023, relative_words_after_2023, region_name)

    return shapiro_pre_result, shapiro_post_result, fig_shapiro_histogram

# Perform the Shapiro-Wilk test for each region (Forschungsfrage 2)
shapiro_pre_eu_question, shapiro_post_eu_question, fig_shapiro_eu_question = process_region_with_shapiro_test(df_eu_question, "EU")
shapiro_pre_asia_question, shapiro_post_asia_question, fig_shapiro_asia_question = process_region_with_shapiro_test(df_asia_question, "Asia")
shapiro_pre_fachhochschule_question, shapiro_post_fachhochschule_question, fig_shapiro_fachhochschule_question = process_region_with_shapiro_test(df_fachhochschule_question, "German universities of applied Science")
shapiro_pre_world_question, shapiro_post_world_question, fig_shapiro_world_question = process_region_with_shapiro_test(df_world_question, "World")
shapiro_pre_germany_question, shapiro_post_germany_question, fig_shapiro_germany_question = process_region_with_shapiro_test(df_world_question, "German Universities")

from scipy.stats import mannwhitneyu

# Function to perform the Mann-Whitney U test
def perform_mann_whitney_test(pre_values, post_values):
    # Perform the Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(pre_values, post_values, alternative='two-sided')
    return u_stat, p_value

# Function to generate a violin plot for Mann-Whitney U test with the results
def generate_mann_whitney_violin_figure(pre_values, post_values, region_name):
    # Create a violin plot for the Pre- and Post-ChatGPT values
    fig = go.Figure()

    # Pre-ChatGPT violin plot
    fig.add_trace(go.Violin(y=pre_values, name="Pre-ChatGPT", box_visible=True, meanline_visible=True, marker_color=color_blue))

    # Post-ChatGPT violin plot
    fig.add_trace(go.Violin(y=post_values, name="Post-ChatGPT", box_visible=True, meanline_visible=True, marker_color=color_orange))

    # Update layout
    fig.update_layout(
        title=f'Mann-Whitney U Test Violin Plot for {region_name}',
        yaxis_title='Relative Frequency',
        legend_title="Time Period"
    )

    return fig

# Function to process the Mann-Whitney U test for a specific region
def process_region_with_mann_whitney_test(df, region_name):
    # Extract question words
    df['Question_Words_Count'] = extract_question_words(df['Question_Words'])

    # Split data into before and after 2023
    df_before_2023 = df[df['PubDate'] <= 2022]
    df_after_2023 = df[df['PubDate'] >= 2023]

    # Aggregate question words
    words_before_2023 = aggregate_question_words(df_before_2023)
    words_after_2023 = aggregate_question_words(df_after_2023)

    # Filter out excluded words
    excluded_words = {'much', 'many', 'far', 'long'}
    words_before_2023_filtered = {k: v for k, v in words_before_2023.items() if k not in excluded_words}
    words_after_2023_filtered = {k: v for k, v in words_after_2023.items() if k not in excluded_words}

    # Calculate relative frequencies
    relative_words_before_2023 = list(calculate_relative_frequencies(words_before_2023_filtered, len(df_before_2023)).values())
    relative_words_after_2023 = list(calculate_relative_frequencies(words_after_2023_filtered, len(df_after_2023)).values())

    # Perform Mann-Whitney U test
    u_stat, p_value = perform_mann_whitney_test(relative_words_before_2023, relative_words_after_2023)

    # Generate violin plot
    fig_violin = generate_mann_whitney_violin_figure(relative_words_before_2023, relative_words_after_2023, region_name)

    return u_stat, p_value, fig_violin

# Perform the Mann-Whitney U test for each region (Forschungsfrage 2)
u_stat_q2_eu, p_value_q2_eu, fig_violin_q2_eu = process_region_with_mann_whitney_test(df_eu_question, "EU")
u_stat_q2_asia, p_value_q2_asia, fig_violin_q2_asia = process_region_with_mann_whitney_test(df_asia_question, "Asia")
u_stat_q2_fachhochschule, p_value_q2_fachhochschule, fig_violin_q2_fachhochschule = process_region_with_mann_whitney_test(df_fachhochschule_question, "German universites of applied Science")
u_stat_q2_world, p_value_q2_world, fig_violin_q2_world = process_region_with_mann_whitney_test(df_world_question, "World")
u_stat_q2_germany, p_value_q2_germany, fig_violin_q2_germany = process_region_with_mann_whitney_test(df_world_question, "German universities")

############################################ End research question 2 ##################################################

######## Code for Research Question 3 ###########################################################################

# Load the regional CSV files
df_eu_sentence = pd.read_csv('Data/Merged_EU_datasets_sentence.csv')
df_asia_sentence = pd.read_csv('Data/Merged_Asia_datasets_sentence.csv')
df_fachhochschule_sentence = pd.read_csv('Data/Merged_FH_datasets_sentence.csv')
df_world_sentence = pd.read_csv('Data/Merged_World_datasets_sentence.csv')
df_germany_sentence = pd.read_csv('Data/Merged_Germany_datasets_sentence.csv')


# Function to split the data into pre- and post-ChatGPT (2022 and earlier vs. 2023 and later)
def split_data(df):
    df['Year'] = pd.to_numeric(df['PubDate'], errors='coerce').fillna(0).astype(int)
    pre_chatgpt = df[df['Year'] <= 2022]
    post_chatgpt = df[df['Year'] >= 2023]
    return pre_chatgpt, post_chatgpt


# Function to calculate average values
def calculate_averages(df):
    avg_sentence_count = df['Sentence_Count'].mean()

    def safe_mean(x):
        try:
            # Convert to list and check if it's valid
            lengths = literal_eval(x)
            if isinstance(lengths, list) and len(lengths) > 0:
                return np.mean(lengths)
            return np.nan
        except (ValueError, SyntaxError):
            return np.nan

    def safe_sum(x):
        try:
            lengths = literal_eval(x)
            if isinstance(lengths, list) and len(lengths) > 0:
                return np.sum(lengths)
            return np.nan
        except (ValueError, SyntaxError):
            return np.nan

    # Calculate average number of words per sentence
    avg_words_per_sentence = df['Sentence_Lengths'].apply(safe_mean).mean()

    # Calculate average number of words per abstract
    avg_words_per_abstract = df['Sentence_Lengths'].apply(safe_sum).mean()

    return avg_sentence_count, avg_words_per_sentence, avg_words_per_abstract


# Visualization using Plotly
def visualize_averages(pre_averages, post_averages, region_name):
    labels = ['Sentence Count', 'Words per Sentence', 'Words per Abstract']

    pre_values = list(pre_averages)
    post_values = list(post_averages)

    x = np.arange(len(labels))
    width = 0.35

    fig = go.Figure()

    # Add bars for Pre-ChatGPT
    fig.add_trace(go.Bar(x=labels, y=pre_values, name='Pre-ChatGPT', marker_color=color_blue))

    # Add bars for Post-ChatGPT
    fig.add_trace(go.Bar(x=labels, y=post_values, name='Post-ChatGPT', marker_color=color_orange))

    # Customize layout
    fig.update_layout(
        title=f"Average Values Comparison (Pre-ChatGPT vs Post-ChatGPT) for {region_name}",
        xaxis_title="Metrics",
        yaxis_title="Average Value",
        barmode='group'
    )

    return fig


# Process and visualize averages for each region
def process_and_visualize_region(df, region_name):
    pre_chatgpt, post_chatgpt = split_data(df)

    # Calculate averages
    pre_averages = calculate_averages(pre_chatgpt)
    post_averages = calculate_averages(post_chatgpt)

    # Generate visualization
    fig = visualize_averages(pre_averages, post_averages, region_name)

    return fig

# Generate figures for each region
fig_eu_avg = process_and_visualize_region(df_eu_sentence, "EU")
fig_asia_avg = process_and_visualize_region(df_asia_sentence, "Asia")
fig_fachhochschule_avg = process_and_visualize_region(df_fachhochschule_sentence, "German universities of applied Science")
fig_world_avg = process_and_visualize_region(df_world_sentence, "World")
fig_germany_avg = process_and_visualize_region(df_germany_sentence, "German universities")

############

def safe_mean(x):
    try:
        lengths = literal_eval(x)
        if isinstance(lengths, list) and len(lengths) > 0:
            valid_lengths = [l for l in lengths if not np.isnan(l)]  # Filter out NaN values
            if len(valid_lengths) > 0:
                return np.mean(valid_lengths)
        return np.nan
    except (ValueError, SyntaxError):
        return np.nan

def safe_sum(x):
    try:
        lengths = literal_eval(x)
        if isinstance(lengths, list) and len(lengths) > 0:
            valid_lengths = [l for l in lengths if not np.isnan(l)]  # Filter out NaN values
            if len(valid_lengths) > 0:
                return np.sum(valid_lengths)
        return np.nan
    except (ValueError, SyntaxError):
        return np.nan

# Filter NaN values before performing Chi-Square tests or any other operations
def filter_valid_data(data):
    return data.dropna()  # Remove NaN values

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Function to perform Kolmogorov-Smirnov test and create histograms with theoretical normal distribution
# Function to perform Kolmogorov-Smirnov test and create histograms with theoretical normal distribution
def ks_test_and_visualization(pre_data, post_data, label, region_name, title_prefix):
    # Drop NaN values and ensure valid data
    pre_data_clean = pre_data.dropna()
    post_data_clean = post_data.dropna()

    if len(pre_data_clean) == 0 or len(post_data_clean) == 0:
        print(f"Warning: No valid data available for {label} in {region_name}. Skipping visualization.")
        return None, f"No valid data for {label} in {region_name}.", f"No valid data for {label} in {region_name}."

    # Perform Kolmogorov-Smirnov test for pre and post-ChatGPT data
    ks_stat_pre, p_value_pre = kstest(pre_data_clean, 'norm', args=(pre_data_clean.mean(), pre_data_clean.std()))
    ks_stat_post, p_value_post = kstest(post_data_clean, 'norm', args=(post_data_clean.mean(), post_data_clean.std()))

    # Create histogram with Plotly
    fig = go.Figure()

    # Pre-ChatGPT histogram and theoretical normal distribution
    fig.add_trace(go.Histogram(x=pre_data_clean, nbinsx=15, histnorm='probability density',
                               name='Pre-ChatGPT', opacity=0.6, marker_color=color_blue))
    x_pre = np.linspace(pre_data_clean.min(), pre_data_clean.max(), 100)
    fig.add_trace(go.Scatter(x=x_pre, y=norm.pdf(x_pre, pre_data_clean.mean(), pre_data_clean.std()),
                             mode='lines', name='Theoretical Normal Distribution (Pre-ChatGPT)',
                             line=dict(color='blue', width=2)))

    # Post-ChatGPT histogram and theoretical normal distribution
    fig.add_trace(go.Histogram(x=post_data_clean, nbinsx=15, histnorm='probability density',
                               name='Post-ChatGPT', opacity=0.6, marker_color=color_orange))
    x_post = np.linspace(post_data_clean.min(), post_data_clean.max(), 100)
    fig.add_trace(go.Scatter(x=x_post, y=norm.pdf(x_post, post_data_clean.mean(), post_data_clean.std()),
                             mode='lines', name='Theoretical Normal Distribution (Post-ChatGPT)',
                             line=dict(color='orange', width=2)))

    # Adjust layout
    fig.update_layout(barmode='overlay', title=f'{title_prefix} for {region_name}',
                      xaxis_title=label, yaxis_title='Density', showlegend=True)

    # Limit the x-axis to a specific range (for example, between 0 and 2000 for "Words per Abstract")
    #fig.update_xaxes(range=[0, 200])  # Adjust this range based on your specific data and expected outliers

    # Interpretation of the results including KS value and p-value
    pre_result_text = (f"Kolmogorov-Smirnov test before ChatGPT: KS-Statistic = {ks_stat_pre:.6f}, "
                       f"p-Value = {p_value_pre:.6f}. ")
    post_result_text = (f"Kolmogorov-Smirnov test after ChatGPT: KS-Statistic = {ks_stat_post:.6f}, "
                        f"p-Value = {p_value_post:.6f}. ")

    pre_result_text += "The data is normally distributed." if p_value_pre > 0.05 else "The data is not normally distributed."
    post_result_text += "The data is normally distributed." if p_value_post > 0.05 else "The data is not normally distributed."

    return fig, pre_result_text, post_result_text

# Function to run Kolmogorov-Smirnov test and create histograms for each data category
def analyze_and_visualize_ks(df, region_name):
    # Split the data into pre-2023 and post-2023 (before and after ChatGPT)
    pre_chatgpt, post_chatgpt = split_data(df)

    # Prepare data for the test and drop NaN values
    pre_sentence_count = pre_chatgpt['Sentence_Count'].dropna()
    post_sentence_count = post_chatgpt['Sentence_Count'].dropna()

    pre_word_lengths = pre_chatgpt['Sentence_Lengths'].apply(
        lambda x: np.mean(literal_eval(x)) if pd.notnull(x) else np.nan).dropna()
    post_word_lengths = post_chatgpt['Sentence_Lengths'].apply(
        lambda x: np.mean(literal_eval(x)) if pd.notnull(x) else np.nan).dropna()

    pre_word_counts = pre_chatgpt['Sentence_Lengths'].apply(
        lambda x: np.sum(literal_eval(x)) if pd.notnull(x) else np.nan).dropna()
    post_word_counts = post_chatgpt['Sentence_Lengths'].apply(
        lambda x: np.sum(literal_eval(x)) if pd.notnull(x) else np.nan).dropna()

    # KS-Test and visualization for each category
    fig_sentence_count, sentence_pre_text, sentence_post_text = ks_test_and_visualization(pre_sentence_count,
                                                                                          post_sentence_count,
                                                                                          "Sentence Count",
                                                                                          region_name,
                                                                                          "Kolmogorov-Smirnov test for average number of sentences per Abstract")
    fig_word_lengths, lengths_pre_text, lengths_post_text = ks_test_and_visualization(pre_word_lengths,
                                                                                      post_word_lengths,
                                                                                      "Words per Sentence",
                                                                                      region_name,
                                                                                      "Kolmogorov-Smirnov test for average number of words per sentence")
    fig_word_counts, counts_pre_text, counts_post_text = ks_test_and_visualization(pre_word_counts,
                                                                                   post_word_counts,
                                                                                   "Words per Abstract",
                                                                                   region_name,
                                                                                   "Kolmogorov-Smirnov test average number of words per Abstract")

    return {
        "figures": {
            "Sentence Count": fig_sentence_count,
            "Words per Sentence": fig_word_lengths,
            "Words per Abstract": fig_word_counts
        },
        "interpretation": {
            "Sentence Count": (sentence_pre_text, sentence_post_text),
            "Words per Sentence": (lengths_pre_text, lengths_post_text),
            "Words per Abstract": (counts_pre_text, counts_post_text)
        }
    }

# Perform Kolmogorov-Smirnov analysis for each region
result_eu_ks = analyze_and_visualize_ks(df_eu_sentence, "EU")
result_asia_ks = analyze_and_visualize_ks(df_asia_sentence, "Asia")
result_fachhochschule_ks = analyze_and_visualize_ks(df_fachhochschule_sentence, "German universities of applied science")
result_world_ks = analyze_and_visualize_ks(df_world_sentence, "World")
result_germany_ks = analyze_and_visualize_ks(df_germany_sentence, "German universities")


# Function to split the data into pre- and post-ChatGPT (2022 and earlier vs. 2023 and later)
def split_data(df):
    df['Year'] = pd.to_numeric(df['PubDate'], errors='coerce').fillna(0).astype(int)
    pre_chatgpt = df[df['Year'] <= 2022]
    post_chatgpt = df[df['Year'] >= 2023]
    return pre_chatgpt, post_chatgpt


# Add 'Words_per_Sentence' and 'Words_per_Abstract' columns to each dataset
df_eu_sentence['Words_per_Sentence'] = df_eu_sentence['Sentence_Lengths'].apply(safe_mean)
df_eu_sentence['Words_per_Abstract'] = df_eu_sentence['Sentence_Lengths'].apply(safe_sum)

df_asia_sentence['Words_per_Sentence'] = df_asia_sentence['Sentence_Lengths'].apply(safe_mean)
df_asia_sentence['Words_per_Abstract'] = df_asia_sentence['Sentence_Lengths'].apply(safe_sum)

df_fachhochschule_sentence['Words_per_Sentence'] = df_fachhochschule_sentence['Sentence_Lengths'].apply(safe_mean)
df_fachhochschule_sentence['Words_per_Abstract'] = df_fachhochschule_sentence['Sentence_Lengths'].apply(safe_sum)

df_world_sentence['Words_per_Sentence'] = df_world_sentence['Sentence_Lengths'].apply(safe_mean)
df_world_sentence['Words_per_Abstract'] = df_world_sentence['Sentence_Lengths'].apply(safe_sum)

df_germany_sentence['Words_per_Sentence'] = df_germany_sentence['Sentence_Lengths'].apply(safe_mean)
df_germany_sentence['Words_per_Abstract'] = df_germany_sentence['Sentence_Lengths'].apply(safe_sum)

# Function to perform Mann-Whitney U-Test and create violin plot with median comparison
def mann_whitney_test_and_visualization(pre_data, post_data, label, region_name):
    # Drop NaN values and ensure valid data
    try:
        pre_data_clean = pre_data.dropna().apply(pd.to_numeric, errors='coerce').dropna()
        post_data_clean = post_data.dropna().apply(pd.to_numeric, errors='coerce').dropna()
    except Exception as e:
        print(f"Error cleaning data for {label} in {region_name}: {e}")
        return None, f"Error processing data for {label} in {region_name}.", f"Error processing data for {label} in {region_name}."

    if len(pre_data_clean) == 0 or len(post_data_clean) == 0:
        print(f"Warning: No valid data available for {label} in {region_name}. Skipping visualization.")
        return None, f"No valid data for {label} in {region_name}.", f"No valid data for {label} in {region_name}."

    # Perform Mann-Whitney U-Test
    try:
        u_stat, p_value = mannwhitneyu(pre_data_clean, post_data_clean, alternative='two-sided')
    except Exception as e:
        print(f"Error performing Mann-Whitney U-Test for {label} in {region_name}: {e}")
        return None, f"Error performing test for {label} in {region_name}.", f"Error performing test for {label} in {region_name}."

    # Create Violin plot with Plotly
    fig = go.Figure()

    # Pre-ChatGPT violin plot
    fig.add_trace(go.Violin(y=pre_data_clean, name=f'Pre-ChatGPT {label}', box_visible=True,
                            meanline_visible=True, line_color=color_blue, fillcolor=color_blue, opacity=0.6))

    # Post-ChatGPT violin plot
    fig.add_trace(go.Violin(y=post_data_clean, name=f'Post-ChatGPT {label}', box_visible=True,
                            meanline_visible=True, line_color=color_orange, fillcolor=color_orange, opacity=0.6))

    # Adjust layout
    fig.update_layout(title=f'Mann-Whitney U test for {region_name}',
                      xaxis_title='Groups', yaxis_title=label, showlegend=False)

    # Interpretation of the results including U statistic and p-value
    result_text = f"{label}: U-Statistic = {u_stat:.6f}, p-Value = {p_value:.6f}. "
    result_text += "The difference is statistically significant." if p_value < 0.05 else "No significant difference detected."

    # Return the figure and the result text
    return fig, result_text

# Split the data for each region into pre- and post-ChatGPT
pre_chatgpt_eu, post_chatgpt_eu = split_data(df_eu_sentence)
pre_chatgpt_asia, post_chatgpt_asia = split_data(df_asia_sentence)
pre_chatgpt_world, post_chatgpt_world = split_data(df_world_sentence)
pre_chatgpt_fachhochschule, post_chatgpt_fachhochschule = split_data(df_fachhochschule_sentence)
pre_chatgpt_germany, post_chatgpt_germany = split_data(df_germany_sentence)

# Mann-Whitney U-Test for each region with the old variable names
# EU Section
fig_eu_mw_sentence_count, result_eu_mw_sentence_count = mann_whitney_test_and_visualization(
    pre_chatgpt_eu['Sentence_Count'], post_chatgpt_eu['Sentence_Count'], "Sentence Count", "EU")

fig_eu_mw_words_per_sentence, result_eu_mw_words_per_sentence = mann_whitney_test_and_visualization(
    pre_chatgpt_eu['Words_per_Sentence'], post_chatgpt_eu['Words_per_Sentence'], "Words per Sentence", "EU")

fig_eu_mw_words_per_abstract, result_eu_mw_words_per_abstract = mann_whitney_test_and_visualization(
    pre_chatgpt_eu['Words_per_Abstract'], post_chatgpt_eu['Words_per_Abstract'], "Words per Abstract", "EU")

# Asia Section
fig_asia_mw_sentence_count, result_asia_mw_sentence_count = mann_whitney_test_and_visualization(
    pre_chatgpt_asia['Sentence_Count'], post_chatgpt_asia['Sentence_Count'], "Sentence Count", "Asia")

fig_asia_mw_words_per_sentence, result_asia_mw_words_per_sentence = mann_whitney_test_and_visualization(
    pre_chatgpt_asia['Words_per_Sentence'], post_chatgpt_asia['Words_per_Sentence'], "Words per Sentence", "Asia")

fig_asia_mw_words_per_abstract, result_asia_mw_words_per_abstract = mann_whitney_test_and_visualization(
    pre_chatgpt_asia['Words_per_Abstract'], post_chatgpt_asia['Words_per_Abstract'], "Words per Abstract", "Asia")

# World Section
fig_world_mw_sentence_count, result_world_mw_sentence_count = mann_whitney_test_and_visualization(
    pre_chatgpt_world['Sentence_Count'], post_chatgpt_world['Sentence_Count'], "Sentence Count", "World")

fig_world_mw_words_per_sentence, result_world_mw_words_per_sentence = mann_whitney_test_and_visualization(
    pre_chatgpt_world['Words_per_Sentence'], post_chatgpt_world['Words_per_Sentence'], "Words per Sentence", "World")

fig_world_mw_words_per_abstract, result_world_mw_words_per_abstract = mann_whitney_test_and_visualization(
    pre_chatgpt_world['Words_per_Abstract'], post_chatgpt_world['Words_per_Abstract'], "Words per Abstract", "World")

# German universities Section
fig_germany_mw_sentence_count, result_germany_mw_sentence_count = mann_whitney_test_and_visualization(
    pre_chatgpt_germany['Sentence_Count'], post_chatgpt_germany['Sentence_Count'], "Sentence Count", "German universities")

fig_germany_mw_words_per_sentence, result_germany_mw_words_per_sentence = mann_whitney_test_and_visualization(
    pre_chatgpt_germany['Words_per_Sentence'], post_chatgpt_germany['Words_per_Sentence'], "Words per Sentence", "German universities")

fig_germany_mw_words_per_abstract, result_germany_mw_words_per_abstract = mann_whitney_test_and_visualization(
    pre_chatgpt_germany['Words_per_Abstract'], post_chatgpt_germany['Words_per_Abstract'], "Words per Abstract", "German universities")

# Fachhochschule Section
fig_fachhochschule_mw_sentence_count, result_fachhochschule_mw_sentence_count = mann_whitney_test_and_visualization(
    pre_chatgpt_fachhochschule['Sentence_Count'], post_chatgpt_fachhochschule['Sentence_Count'], "Sentence Count", "German universities of applied science")

fig_fachhochschule_mw_words_per_sentence, result_fachhochschule_mw_words_per_sentence = mann_whitney_test_and_visualization(
    pre_chatgpt_fachhochschule['Words_per_Sentence'], post_chatgpt_fachhochschule['Words_per_Sentence'], "Words per Sentence", "German universities of applied science")

fig_fachhochschule_mw_words_per_abstract, result_fachhochschule_mw_words_per_abstract = mann_whitney_test_and_visualization(
    pre_chatgpt_fachhochschule['Words_per_Abstract'], post_chatgpt_fachhochschule['Words_per_Abstract'], "Words per Abstract", "German universities of applied science")


############################################ End research question 3 ##################################################


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
        # Leere Zeile für zusätzlichen Abstand
        dbc.Row([dbc.Col(html.Div(), style={"height": "30px"})]),

        dbc.Row([
            dbc.Col(html.H2("Research Question 1: Change in the use of certain words"), width=6),
            dbc.Col(html.P(
                "How has the use of certain words in scientific papers changed since the introduction of ChatGPT?.")),
        ]),


        dbc.Row([dbc.Col(html.Div(), style={"height": "30px"})]),

        dbc.Row([
            dcc.Dropdown(
                options=[
                    {'label': 'German universities', 'value': 'german_universities'},
                    {'label': 'German universities of applied Science', 'value': 'german_fachhochschulen'},
                    {'label': 'EU universities', 'value': 'eu_universities'},
                    {'label': 'Asia universities', 'value': 'asia_universities'},
                    {'label': 'World universities', 'value': 'world_universities'},
                ],
                placeholder='Select your Topic',
                value='german_universities',
                id='RQ1_Dropdown'
            )
        ]),
        # Container für die Diagramme
        html.Div(id='charts-container-RQ1'),

        dbc.Row([dbc.Col(html.H2(""), className="mb-4 text-center")]),

        # Research Question 1: Changes in Word Usage (using real data)

        # Research Question 2

        dbc.Row([
            dbc.Col(html.H2("Research Question 2: Change in the use of Question words and Question Marks"), width=6),
            dbc.Col(html.P(
                "What changes have there been in the use of question words (e.g. what, why) in scientific papers since the introduction of ChatGPT?.")),
        ]),
        # Leere Zeile für zusätzlichen Abstand
        dbc.Row([dbc.Col(html.Div(), style={"height": "30px"})]),

        dbc.Row([
            dcc.Dropdown(
                options=[
                    {'label': 'German universities', 'value': 'german_universities'},
                    {'label': 'German universities of applied Science', 'value': 'german_fachhochschulen'},
                    {'label': 'EU universities', 'value': 'eu_universities'},
                    {'label': 'Asia universities', 'value': 'asia_universities'},
                    {'label': 'World universities', 'value': 'world_universities'},
                ],
                placeholder='Select your Topic',
                value='german_universities',
                id='RQ2_Dropdown'
            )
        ]),

        html.Div(id='charts-container-RQ2'),


        # Research Question 3: Sentence Length in Scientific Papers
        dbc.Row([
            dbc.Col(html.H2("Research Question 3: Sentence Length in Scientific Papers"), width=6),
            dbc.Col(html.P(
                "How has the length of sentences in scientific papers changed since the introduction of ChatGPT? We aim to investigate whether sentences have become longer or shorter post-ChatGPT.")),
        ]),
        # Leere Zeile für zusätzlichen Abstand
        dbc.Row([dbc.Col(html.Div(), style={"height": "30px"})]),

        dbc.Row([
            dcc.Dropdown(
                options=[
                    {'label': 'German universities', 'value': 'german_universities'},
                    {'label': 'German universities of applied Science', 'value': 'german_fachhochschulen'},
                    {'label': 'EU universities', 'value': 'eu_universities'},
                    {'label': 'Asia universities', 'value': 'asia_universities'},
                    {'label': 'World universities', 'value': 'world_universities'},
                ],
                placeholder='Select your Topic',
                value='german_universities',
                id='RQ3_Dropdown'
            )
        ]),
        html.Div(id='charts-container-RQ3'),


        # Research Question 4: Comparison of Flagged Keywords in PDFs and Abstracts
        dbc.Row([
            dbc.Col(html.H4("Research Question 4: Comparison of PDF and abstract flagging"), width=6),
            dbc.Col(html.P(
                "How often do certain keywords get flagged in papers?"
                "Is there a correlation between flagged keywords in abstracts and pdf files?")),
        ]),
        dbc.Row([
            html.Div(id='slider-RQ4', children=
            [dcc.Slider(id='RQ4-year-slider',
                        min=RQ_4_df['Year'].min(),
                        max=RQ_4_df['Year'].max(),
                        value=RQ_4_df['Year'].min(),
                        marks={str(year): str(year) for year in RQ_4_df['Year'].unique()},
                        step=None
                        )], style={'width': '50%', 'display': 'inline-block'}),
            # inline-block : to show slider and dropdown in the same line

            html.Div(id='RQ4-radio', children=
            [dcc.RadioItems(id='RQ4-Keyword-radio',
                            options=[
                                {'label': 'PDF', 'value': 'Count_PDF'},
                                {'label': 'Abstract', 'value': 'Count_Abs'}],
                            value='Count_PDF',
                            )], style={'width': '50%', 'display': 'inline-block'}),

            html.Div(id='RQ4-radio-log', children=
            [dcc.RadioItems(id='RQ4-Keyword-radio-log',
                            options=[
                                {'label': 'Log scale', 'value': True},
                                {'label': 'Linear scale', 'value': False}],
                            value=True,
                            )], style={'width': '50%', 'display': 'inline-block'}),

            dcc.Graph(id='RQ4-barchart'),
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
            dbc.Col(dcc.Graph(
                id="comparison-graph-5",
                figure=px.bar(df_grouped_5, x='type', y='percentage', color='flag',
                           title="Flagged vs Non-Flagged Papers (CAU vs Other Universities)",
                           labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers',
                                   'flag': 'Flagged Status'},
                           barmode='stack')

            )),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Explanation"),
                dbc.CardBody(html.P(
                    "This bar chart compares the percentage of flagged papers from Christian-Albrechts-Universität zu Kiel (CAU) "
                    "with other universities. Flagged papers are those that contain certain keywords associated with being influenced "
                    "by ChatGPT. Non-flagged papers are those without these keywords. "
                    "The proportions are similar for both CAU and other universities, indicating comparable levels of ChatGPT-influenced content.",
                    style={"color": "#444", "font-size": "15px"}
                ))
            ], color="light", outline=True), width=12),
        ], className="mb-5"),

        # Comparison Before and After 1.1.2023 for Research Question 5
        dbc.Row([
            dbc.Col(html.H4(
                "Research Question 5: Comparison of Papers Before and After 1.1.2023 (CAU vs Other Universities)"),
                    width=6),
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
                              labels={'type': 'Institution Type', 'percentage': 'Percentage of Papers',
                                      'DateCategory': 'Publication Date Category'})
            )),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Explanation"),
                dbc.CardBody(html.P(
                    "This chart compares the percentage of flagged papers before and after January 1, 2023, when ChatGPT became widely available. "
                    "The graph shows an increase in flagged papers after 1.1.2023, especially for other universities, suggesting the rising influence of ChatGPT.",
                    "The proportions are similar for both CAU and other universities, indicating comparable levels of ChatGPT-influenced content.",
                    style={"color": "#444", "font-size": "15px"}
                ))
            ], color="light", outline=True), width=12),
        ], className="mb-5"),

        # Line graph for 2020-2024 flagged papers (Research Question 5)
        dbc.Row([
            dbc.Col(html.H4("Research Question 5: Comparing CAU with other German Universities"), width=6),
            dbc.Col(html.P(
                "This graph shows the percentage of flagged papers from 2012 to 2024, separated by institution type (CAU and other universities).")),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(
                id="year-comparison-graph-5-with-trend",
                figure=px.line(df_year_grouped_5, x='PubDate', y='percentage', color='type',
                               color_discrete_sequence=colors_blind_friendly,
                               title=title_with_p_value,  # Add p-value to the title
                               labels={'PubDate': 'Publication Year', 'percentage': 'Percentage of Papers',
                                       'type': 'Institution Type'})
                .add_scatter(x=df_year_grouped_5['PubDate'], y=df_year_grouped_5['trend'], mode='lines',
                             name='Trend Line')
                .update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Set to display full years on the x-axis
                # Set to display whole years on the x-axis
            )),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Explanation"),
                dbc.CardBody(html.P(
                    "This line chart tracks the percentage of flagged papers from 2012 to 2024, comparing CAU with other universities. "
                    "The trend line indicates a general upward trajectory, suggesting a growing influence of ChatGPT on scientific papers over these years.",
                    "We see no significant difference between CAU and other universities, as indicated by the p-value.",
                    style={"color": "#444", "font-size": "15px"}
                ))
            ], color="light", outline=True), width=12),
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

        # Interactive description for Research Question 7
        dbc.Row([dbc.Col(html.H4(
            "Research Question 7: How do the influences of ChatGPT on scientific papers differ between universities worldwide?"),
                         width=12)]),
        dbc.Row([dbc.Col(html.P(
            "Now you can explore this question interactively! Using the dropdown menu below, you can select combinations "
            "of different datasets (e.g., German Universities + Fachhochschule, EU + Asia) and compare the flagged papers between "
            "these institutions. Once you select a combination, two graphs will appear: a bar chart and a line chart displaying trends over time. "
            "By selecting different combinations, you can see how the influence of ChatGPT varies across regions and institutions. "
            "Feel free to experiment and discover insights!"
        ), width=12)]),
        dbc.Row([dbc.Col(dcc.Dropdown(
            id='dataset-combination-dropdown',
            options=[
                {'label': 'German University + Fachhochschule', 'value': 'df_combined_Uni_FH'},
                {'label': 'Germany + EU', 'value': 'df_combined_Germany_EU'},
                {'label': 'Germany + Asia', 'value': 'df_combined_Germany_Asia'},
                {'label': 'Germany + World', 'value': 'df_combined_Germany_World'},
                {'label': 'EU + Asia', 'value': 'df_combined_EU_Asia'},
                {'label': 'EU + World', 'value': 'df_combined_EU_World'},
                {'label': 'Asia + World', 'value': 'df_combined_Asia_World'}
            ],
            value='df_combined_EU_World',  # Default selection
            clearable=False
        ), width=12)]),

        dbc.Row([dbc.Col(dcc.Graph(id='comparison-bar-chart7'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='comparison-line-chart7'), width=12)]),
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
