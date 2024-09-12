import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from layout import homepage, projects_section, about_section
from dash import Input, Output
from layout import *  # Importiert alle Variablen, Funktionen und Klassen aus layout.py

# Initialize Dash app with external Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Define the layout of the app, linking the sections with smooth scrolling
app.layout = html.Div([
    # Navbar for smooth navigation between sections
    dbc.NavbarSimple(
        children=[
            #dbc.NavItem(dbc.NavLink("Home", href="#start")),
        ],
        brand="Joule im Pool",
        brand_href="#start",
        color="dark",
        dark=True,
        className="mb-4",
    ),

    # Home, Projects, About sections
    homepage,
    projects_section,
    about_section,
])

@app.callback(
    Output('charts-container-RQ1', 'children'),
    #Input('RQ1_Dropdown', 'value')
    [Input('RQ1_Dropdown', 'value')]
)
def update_charts(region):
    if region == 'german_universities':
        return [
            dbc.Row([
                dbc.Col(html.H2("German universities Section"), className="mb-4 text-center",
                        style={"margin-top": "20px"})
            ]),
            # Bar Chart
            dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-universities", figure=fig_bar_universities))], className="mb-5"),
            # Chi-Square Heatmap
            dbc.Row([dbc.Col(dcc.Graph(id="chi-square-heatmap-universities", figure=fig_heatmap_universities))], className="mb-5"),
            # Chi-Square Test Results
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for German Universities: chi2 = {chi2_universities:.6f}, p-value = {p_universities:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected, although individual words may vary in their contributions.' if p_universities < 0.05 else 'No significant difference was found cumulatively across all words, indicating that the observed word usage does not differ significantly from what was expected, although individual words may vary in their contributions.'}"
            ))], className="mb-5"),
            # Shapiro-Wilk Histogram
            dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-universities", figure=fig_shapiro_universities))], className="mb-5"),
            # Shapiro-Wilk Test Results
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT German Universities: p-value = {shapiro_pre_universities[1]:.6f}, stat = {shapiro_pre_universities[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_universities[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT German Universities: p-value = {shapiro_post_universities[1]:.6f}, stat = {shapiro_post_universities[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_universities[1] < 0.05 else 'Normally distributed'}."
                )),
            ], className="mb-5"),
            # Mann-Whitney U Test Violin Plot
            dbc.Row([dbc.Col(dcc.Graph(id="mann-whitney-violin-universities", figure=fig_violin_universities))], className="mb-5"),
            # Mann-Whitney U Test Results
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for German Universities: U-statistic = {u_stat_universities:.6f}, p-value = {p_value_universities:.6f}. "
                f"{'Significant difference' if p_value_universities < 0.05 else 'No significant difference'}."
            ))], className="mb-5")
        ]
    elif region == 'german_fachhochschulen':
        return [
            dbc.Row([dbc.Col(html.H2("German Universities of Applied Sciences Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            # Bar Chart
            dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-fachhochschulen", figure=fig_bar_fachhochschulen))], className="mb-5"),
            # Chi-Square Heatmap
            dbc.Row([dbc.Col(dcc.Graph(id="chi-square-heatmap-fachhochschulen", figure=fig_heatmap_fachhochschulen))], className="mb-5"),
            # Chi-Square Test Results
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for German Universities of Applied Sciences: chi2 = {chi2_fachhochschulen:.6f}, p-value = {p_fachhochschulen:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected, although individual words may vary in their contributions.' if p_fachhochschulen < 0.05 else 'No significant difference was found cumulatively across all words, indicating that the observed word usage does not differ significantly from what was expected, although individual words may vary in their contributions.'}"
            ))], className="mb-5"),
            # Shapiro-Wilk Histogram
            dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-fachhochschulen", figure=fig_shapiro_fachhochschulen))], className="mb-5"),
            # Shapiro-Wilk Test Results
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT German Universities of Applied Sciences: p-value = {shapiro_pre_fachhochschulen[1]:.6f}, stat = {shapiro_pre_fachhochschulen[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_fachhochschulen[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT German Universities of Applied Sciences: p-value = {shapiro_post_fachhochschulen[1]:.6f}, stat = {shapiro_post_fachhochschulen[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_fachhochschulen[1] < 0.05 else 'Normally distributed'}."
                )),
            ], className="mb-5"),
            # Mann-Whitney U Test Violin Plot
            dbc.Row([dbc.Col(dcc.Graph(id="mann-whitney-violin-fachhochschulen", figure=fig_violin_fachhochschulen))], className="mb-5"),
            # Mann-Whitney U Test Results
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for German Universities of Applied Sciences: U-statistic = {u_stat_fachhochschulen:.6f}, p-value = {p_value_fachhochschulen:.6f}. "
                f"{'Significant difference' if p_value_fachhochschulen < 0.05 else 'No significant difference'}."
            ))], className="mb-5")
        ]
    elif region == 'eu_universities':
        return [
            dbc.Row([dbc.Col(html.H2("EU Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            # Bar Chart
            dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-eu", figure=fig_bar_eu))], className="mb-5"),
            # Chi-Square Heatmap
            dbc.Row([dbc.Col(dcc.Graph(id="chi-square-heatmap-eu", figure=fig_heatmap_eu))], className="mb-5"),
            # Chi-Square Test Results
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for EU Universities: chi2 = {chi2_eu:.6f}, p-value = {p_eu:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected, although individual words may vary in their contributions.' if p_eu < 0.05 else 'No significant difference was found cumulatively across all words, indicating that the observed word usage does not differ significantly from what was expected, although individual words may vary in their contributions.'}"
            ))], className="mb-5"),
            # Shapiro-Wilk Histogram
            dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-eu", figure=fig_shapiro_eu))], className="mb-5"),
            # Shapiro-Wilk Test Results
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT EU Universities: p-value = {shapiro_pre_eu[1]:.6f}, stat = {shapiro_pre_eu[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_eu[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT EU Universities: p-value = {shapiro_post_eu[1]:.6f}, stat = {shapiro_post_eu[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_eu[1] < 0.05 else 'Normally distributed'}."
                )),
            ], className="mb-5"),
            # Mann-Whitney U Test Violin Plot
            dbc.Row([dbc.Col(dcc.Graph(id="mann-whitney-violin-eu", figure=fig_violin_eu))], className="mb-5"),
            # Mann-Whitney U Test Results
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for EU Universities: U-statistic = {u_stat_eu:.6f}, p-value = {p_value_eu:.6f}. "
                f"{'Significant difference' if p_value_eu < 0.05 else 'No significant difference'}."
            ))], className="mb-5")
        ]
    elif region == 'asia_universities':
        return [
            dbc.Row([dbc.Col(html.H2("Asia Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            # Bar Chart
            dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-asia", figure=fig_bar_asia))], className="mb-5"),
            # Chi-Square Heatmap
            dbc.Row([dbc.Col(dcc.Graph(id="chi-square-heatmap-asia", figure=fig_heatmap_asia))], className="mb-5"),
            # Chi-Square Test Results
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for Asia Universities: chi2 = {chi2_asia:.6f}, p-value = {p_asia:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected, although individual words may vary in their contributions.' if p_asia < 0.05 else 'No significant difference was found cumulatively across all words, indicating that the observed word usage does not differ significantly from what was expected, although individual words may vary in their contributions.'}"
            ))], className="mb-5"),
            # Shapiro-Wilk Histogram
            dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-asia", figure=fig_shapiro_asia))], className="mb-5"),
            # Shapiro-Wilk Test Results
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT Asia Universities: p-value = {shapiro_pre_asia[1]:.6f}, stat = {shapiro_pre_asia[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_asia[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT Asia Universities: p-value = {shapiro_post_asia[1]:.6f}, stat = {shapiro_post_asia[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_asia[1] < 0.05 else 'Normally distributed'}."
                )),
            ], className="mb-5"),
            # Mann-Whitney U Test Violin Plot
            dbc.Row([dbc.Col(dcc.Graph(id="mann-whitney-violin-asia", figure=fig_violin_asia))], className="mb-5"),
            # Mann-Whitney U Test Results
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for Asia Universities: U-statistic = {u_stat_asia:.6f}, p-value = {p_value_asia:.6f}. "
                f"{'Significant difference' if p_value_asia < 0.05 else 'No significant difference'}."
            ))], className="mb-5")
        ]
    elif region == 'world_universities':
        return [
            dbc.Row([dbc.Col(html.H2("World universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            # Bar Chart
            dbc.Row([dbc.Col(dcc.Graph(id="word-usage-bar-world", figure=fig_bar_world_q1))], className="mb-5"),
            # Chi-Square Heatmap
            dbc.Row([dbc.Col(dcc.Graph(id="chi-square-heatmap-world", figure=fig_heatmap_world_q1))], className="mb-5"),
            # Chi-Square Test Results
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for World universities: chi2 = {chi2_world_q1:.6f}, p-value = {p_world_q1:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected, although individual words may vary in their contributions.' if p_world_q1 < 0.05 else 'No significant difference was found cumulatively across all words, indicating that the observed word usage does not differ significantly from what was expected, although individual words may vary in their contributions.'}"
            ))], className="mb-5"),
            # Shapiro-Wilk Histogram
            dbc.Row([dbc.Col(dcc.Graph(id="shapiro-histogram-world", figure=fig_shapiro_world_q1))], className="mb-5"),
            # Shapiro-Wilk Test Results
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT World universities: p-value = {shapiro_pre_world_q1[1]:.6f}, stat = {shapiro_pre_world_q1[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_world_q1[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT World universities: p-value = {shapiro_post_world_q1[1]:.6f}, stat = {shapiro_post_world_q1[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_world_q1[1] < 0.05 else 'Normally distributed'}."
                )),
            ], className="mb-5"),
            # Mann-Whitney U Test Violin Plot
            dbc.Row([dbc.Col(dcc.Graph(id="mann-whitney-violin-world", figure=fig_violin_world_q1))], className="mb-5"),
            # Mann-Whitney U Test Results
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for World universities: U-statistic = {u_stat_world_q1:.6f}, p-value = {p_value_world_q1:.6f}. "
                f"{'Significant difference' if p_value_world_q1 < 0.05 else 'No significant difference'}."
            ))], className="mb-5")
        ]


@app.callback(
    Output('charts-container-RQ2', 'children'),
    [Input('RQ2_Dropdown', 'value')]
)
def update_charts_RQ2(region):

    if region == 'german_universities':
        return [
            dbc.Row([dbc.Col(html.H2("German Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_words_germany))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_marks_germany))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_heatmap_q2_germany))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for German universities: chi2 = {chi2_q2_germany:.6f}, p-value = {p_q2_germany:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected.' if p_q2_germany < 0.05 else 'No significant difference was found.'}"
            ))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_shapiro_germany_question))], className="mb-5"),
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT German universities: p-value = {shapiro_pre_germany_question[1]:.6f}, stat = {shapiro_pre_germany_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_germany_question[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT German universities: p-value = {shapiro_post_germany_question[1]:.6f}, stat = {shapiro_post_germany_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_germany_question[1] < 0.05 else 'Normally distributed'}."
                ))
            ], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_violin_q2_germany))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for German universities: U-statistic = {u_stat_q2_germany:.6f}, p-value = {p_value_q2_germany:.6f}. "
                f"{'Significant difference' if p_value_q2_germany < 0.05 else 'No significant difference'}."
            ))], className="mb-5"),
        ]
    elif region == 'german_fachhochschulen':
        return [
            dbc.Row([dbc.Col(html.H2("German Universities of Applied Sciences Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_words_fachhochschule))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_marks_fachhochschule))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_heatmap_q2_fachhochschule))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for German Universities of Applied Sciences: chi2 = {chi2_q2_fachhochschule:.6f}, p-value = {p_q2_fachhochschule:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected.' if p_q2_fachhochschule < 0.05 else 'No significant difference was found.'}"
            ))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_shapiro_fachhochschule_question))], className="mb-5"),
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT Fachhochschule: p-value = {shapiro_pre_fachhochschule_question[1]:.6f}, stat = {shapiro_pre_fachhochschule_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_fachhochschule_question[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT Fachhochschule: p-value = {shapiro_post_fachhochschule_question[1]:.6f}, stat = {shapiro_post_fachhochschule_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_fachhochschule_question[1] < 0.05 else 'Normally distributed'}."
                ))
            ], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_violin_q2_fachhochschule))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for German Universities of Applied Sciences: U-statistic = {u_stat_q2_fachhochschule:.6f}, p-value = {p_value_q2_fachhochschule:.6f}. "
                f"{'Significant difference' if p_value_q2_fachhochschule < 0.05 else 'No significant difference'}."
            ))], className="mb-5"),
        ]
    elif region == 'eu_universities':
        return [
            dbc.Row([dbc.Col(html.H2("EU Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_words_eu))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_marks_eu))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_heatmap_q2_eu))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for EU Universities: chi2 = {chi2_q2_eu:.6f}, p-value = {p_q2_eu:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected.' if p_q2_eu < 0.05 else 'No significant difference was found.'}"
            ))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_shapiro_eu_question))], className="mb-5"),
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT EU: p-value = {shapiro_pre_eu_question[1]:.6f}, stat = {shapiro_pre_eu_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_eu_question[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT EU: p-value = {shapiro_post_eu_question[1]:.6f}, stat = {shapiro_post_eu_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_eu_question[1] < 0.05 else 'Normally distributed'}."
                ))
            ], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_violin_q2_eu))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for EU Universities: U-statistic = {u_stat_q2_eu:.6f}, p-value = {p_value_q2_eu:.6f}. "
                f"{'Significant difference' if p_value_q2_eu < 0.05 else 'No significant difference'}."
            ))], className="mb-5"),
        ]
    elif region == 'asia_universities':
        return [
            dbc.Row([dbc.Col(html.H2("Asia Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_words_asia))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_marks_asia))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_heatmap_q2_asia))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for Asia Universities: chi2 = {chi2_q2_asia:.6f}, p-value = {p_q2_asia:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected.' if p_q2_asia < 0.05 else 'No significant difference was found.'}"
            ))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_shapiro_asia_question))], className="mb-5"),
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT Asia: p-value = {shapiro_pre_asia_question[1]:.6f}, stat = {shapiro_pre_asia_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_asia_question[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT Asia: p-value = {shapiro_post_asia_question[1]:.6f}, stat = {shapiro_post_asia_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_asia_question[1] < 0.05 else 'Normally distributed'}."
                ))
            ], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_violin_q2_asia))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for Asia Universities: U-statistic = {u_stat_q2_asia:.6f}, p-value = {p_value_q2_asia:.6f}. "
                f"{'Significant difference' if p_value_q2_asia < 0.05 else 'No significant difference'}."
            ))], className="mb-5"),
        ]
    elif region == 'world_universities':
        return [
            dbc.Row([dbc.Col(html.H2("World Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_words_world))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_marks_world))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_heatmap_q2_world))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Chi-Square Test for World Universities: chi2 = {chi2_q2_world:.6f}, p-value = {p_q2_world:.6f}. "
                f"{'A significant difference was found cumulatively across all words, indicating that the observed word usage differs significantly from what was expected.' if p_q2_world < 0.05 else 'No significant difference was found.'}"
            ))], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_shapiro_world_question))], className="mb-5"),
            dbc.Row([
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Pre-ChatGPT World: p-value = {shapiro_pre_world_question[1]:.6f}, stat = {shapiro_pre_world_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_pre_world_question[1] < 0.05 else 'Normally distributed'}."
                )),
                dbc.Col(html.P(
                    f"Shapiro-Wilk Test for Post-ChatGPT World: p-value = {shapiro_post_world_question[1]:.6f}, stat = {shapiro_post_world_question[0]:.6f}. "
                    f"{'Not normally distributed' if shapiro_post_world_question[1] < 0.05 else 'Normally distributed'}."
                ))
            ], className="mb-5"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_violin_q2_world))], className="mb-5"),
            dbc.Row([dbc.Col(html.P(
                f"Mann-Whitney U Test for World Universities: U-statistic = {u_stat_q2_world:.6f}, p-value = {p_value_q2_world:.6f}. "
                f"{'Significant difference' if p_value_q2_world < 0.05 else 'No significant difference'}."
            ))], className="mb-5"),
        ]

    # Fallback für nicht erkannte Regionen
    return html.P("No region selected for Question Words (Research Question 2)")

@app.callback(
    Output('charts-container-RQ3', 'children'),
    [Input('RQ3_Dropdown', 'value')]
)
def update_charts_RQ3(region):
    dbc.Row([dbc.Col(html.Div(), style={"height": "30px"})]),
    if region == 'german_universities':
        return [
            dbc.Row([dbc.Col(html.H2("German universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_germany_avg))], className="mb-5"),

            # Kolmogorov-Smirnov Test for Sentence Count in German universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_germany_ks["figures"]["Sentence Count"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_ks['interpretation']['Sentence Count'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_ks['interpretation']['Sentence Count'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Sentence in German universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_germany_ks["figures"]["Words per Sentence"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_ks['interpretation']['Words per Sentence'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_ks['interpretation']['Words per Sentence'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Abstract in German universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_germany_ks["figures"]["Words per Abstract"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_ks['interpretation']['Words per Abstract'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_ks['interpretation']['Words per Abstract'][1]}"))]),

            # Mann-Whitney Test for Sentence Count in German universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_germany_mw_sentence_count))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_mw_sentence_count}"))]),

            # Mann-Whitney Test for Words per Sentence in German universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_germany_mw_words_per_sentence))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_mw_words_per_sentence}"))]),

            # Mann-Whitney Test for Words per Abstract in German universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_germany_mw_words_per_abstract))]),
            dbc.Row([dbc.Col(html.P(f"{result_germany_mw_words_per_abstract}"))]),
        ]
    elif region == 'eu_universities':
        return [
            dbc.Row([dbc.Col(html.H2("EU Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_eu_avg))], className="mb-5"),

            # Kolmogorov-Smirnov Test for Sentence Count in EU universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_eu_ks["figures"]["Sentence Count"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_ks['interpretation']['Sentence Count'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_ks['interpretation']['Sentence Count'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Sentence in EU universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_eu_ks["figures"]["Words per Sentence"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_ks['interpretation']['Words per Sentence'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_ks['interpretation']['Words per Sentence'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Abstract in EU universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_eu_ks["figures"]["Words per Abstract"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_ks['interpretation']['Words per Abstract'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_ks['interpretation']['Words per Abstract'][1]}"))]),

            # Mann-Whitney Test for Sentence Count in EU universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_eu_mw_sentence_count))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_mw_sentence_count}"))]),

            # Mann-Whitney Test for Words per Sentence in EU universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_eu_mw_words_per_sentence))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_mw_words_per_sentence}"))]),

            # Mann-Whitney Test for Words per Abstract in EU universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_eu_mw_words_per_abstract))]),
            dbc.Row([dbc.Col(html.P(f"{result_eu_mw_words_per_abstract}"))]),
        ]
    elif region == 'asia_universities':
        return [
            dbc.Row([dbc.Col(html.H2("Asia Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_asia_avg))], className="mb-5"),

            # Kolmogorov-Smirnov Test for Sentence Count in Asia universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_asia_ks["figures"]["Sentence Count"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_ks['interpretation']['Sentence Count'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_ks['interpretation']['Sentence Count'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Sentence in Asia universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_asia_ks["figures"]["Words per Sentence"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_ks['interpretation']['Words per Sentence'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_ks['interpretation']['Words per Sentence'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Abstract in Asia universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_asia_ks["figures"]["Words per Abstract"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_ks['interpretation']['Words per Abstract'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_ks['interpretation']['Words per Abstract'][1]}"))]),

            # Mann-Whitney Test for Sentence Count in Asia universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_asia_mw_sentence_count))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_mw_sentence_count}"))]),

            # Mann-Whitney Test for Words per Sentence in Asia universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_asia_mw_words_per_sentence))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_mw_words_per_sentence}"))]),

            # Mann-Whitney Test for Words per Abstract in Asia universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_asia_mw_words_per_abstract))]),
            dbc.Row([dbc.Col(html.P(f"{result_asia_mw_words_per_abstract}"))]),
        ]
    elif region == 'world_universities':
        return [
            dbc.Row([dbc.Col(html.H2("World Universities Section"), className="mb-4 text-center",  style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_world_avg))], className="mb-5"),

            # Kolmogorov-Smirnov Test for Sentence Count in World universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_world_ks["figures"]["Sentence Count"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_ks['interpretation']['Sentence Count'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_ks['interpretation']['Sentence Count'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Sentence in World universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_world_ks["figures"]["Words per Sentence"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_ks['interpretation']['Words per Sentence'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_ks['interpretation']['Words per Sentence'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Abstract in World universities
            dbc.Row([dbc.Col(dcc.Graph(figure=result_world_ks["figures"]["Words per Abstract"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_ks['interpretation']['Words per Abstract'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_ks['interpretation']['Words per Abstract'][1]}"))]),

            # Mann-Whitney Test for Sentence Count in World universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_world_mw_sentence_count))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_mw_sentence_count}"))]),

            # Mann-Whitney Test for Words per Sentence in World universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_world_mw_words_per_sentence))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_mw_words_per_sentence}"))]),

            # Mann-Whitney Test for Words per Abstract in World universities
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_world_mw_words_per_abstract))]),
            dbc.Row([dbc.Col(html.P(f"{result_world_mw_words_per_abstract}"))]),
        ]
    elif region == 'german_fachhochschulen':
        return [
            dbc.Row([dbc.Col(html.H2("German universities of applied Science Section"), className="mb-4 text-center", style={"margin-top": "20px"})]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_fachhochschule_avg))], className="mb-5"),

            # Kolmogorov-Smirnov Test for Sentence Count in Fachhochschule
            dbc.Row([dbc.Col(dcc.Graph(figure=result_fachhochschule_ks["figures"]["Sentence Count"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_ks['interpretation']['Sentence Count'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_ks['interpretation']['Sentence Count'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Sentence in Fachhochschule
            dbc.Row([dbc.Col(dcc.Graph(figure=result_fachhochschule_ks["figures"]["Words per Sentence"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_ks['interpretation']['Words per Sentence'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_ks['interpretation']['Words per Sentence'][1]}"))]),

            # Kolmogorov-Smirnov Test for Words per Abstract in Fachhochschule
            dbc.Row([dbc.Col(dcc.Graph(figure=result_fachhochschule_ks["figures"]["Words per Abstract"]))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_ks['interpretation']['Words per Abstract'][0]}"))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_ks['interpretation']['Words per Abstract'][1]}"))]),

            # Mann-Whitney Test for Sentence Count in Fachhochschule
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_fachhochschule_mw_sentence_count))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_mw_sentence_count}"))]),

            # Mann-Whitney Test for Words per Sentence in Fachhochschule
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_fachhochschule_mw_words_per_sentence))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_mw_words_per_sentence}"))]),

            # Mann-Whitney Test for Words per Abstract in Fachhochschule
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_fachhochschule_mw_words_per_abstract))]),
            dbc.Row([dbc.Col(html.P(f"{result_fachhochschule_mw_words_per_abstract}"))]),
        ]

    return html.P("No region selected for the third Research Question")

if __name__ == '__main__':
    #app.run(debug=True)
    #app.run_server(debug=True)
    app.run(debug=True)
