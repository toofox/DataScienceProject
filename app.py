import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from layout import homepage, projects_section, about_section, task_workflow
import pandas as pd

# Initialize Dash app with external Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server
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

# 1. Combine German University and Fachhochschul
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
    [Output('comparison-bar-chart7', 'figure'),
     Output('comparison-line-chart7', 'figure')],
    Input('dataset-combination-dropdown', 'value')
)
def update_combination_graph(selected_combination):
    # Create combinations inside the callback to ensure fresh data on each call
    if selected_combination == 'df_combined_Uni_FH':
        combined_df = pd.concat([df_german_uni, df_german_fh])
    elif selected_combination == 'df_combined_Uni_Germany':
        combined_df = pd.concat([df_german_uni, df_germany])
    elif selected_combination == 'df_combined_Uni_EU':
        combined_df = pd.concat([df_german_uni, df_eu])
    elif selected_combination == 'df_combined_Uni_Asia':
        combined_df = pd.concat([df_german_uni, df_asia])
    elif selected_combination == 'df_combined_Uni_World':
        combined_df = pd.concat([df_german_uni, df_world])
    elif selected_combination == 'df_combined_FH_Germany':
        combined_df = pd.concat([df_german_fh, df_germany])
    elif selected_combination == 'df_combined_FH_EU':
        combined_df = pd.concat([df_german_fh, df_eu])
    elif selected_combination == 'df_combined_FH_Asia':
        combined_df = pd.concat([df_german_fh, df_asia])
    elif selected_combination == 'df_combined_FH_World':
        combined_df = pd.concat([df_german_fh, df_world])
    elif selected_combination == 'df_combined_Germany_EU':
        combined_df = pd.concat([df_germany, df_eu])
    elif selected_combination == 'df_combined_Germany_Asia':
        combined_df = pd.concat([df_germany, df_asia])
    elif selected_combination == 'df_combined_Germany_World':
        combined_df = pd.concat([df_germany, df_world])
    elif selected_combination == 'df_combined_EU_Asia':
        combined_df = pd.concat([df_eu, df_asia])
    elif selected_combination == 'df_combined_EU_World':
        combined_df = pd.concat([df_eu, df_world])
    elif selected_combination == 'df_combined_Asia_World':
        combined_df = pd.concat([df_asia, df_world])

    # Generate both the bar and line charts using the task_workflow
    fig_bar, fig_line = task_workflow(combined_df, [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], "Selected Combination")

    # Return both figures to update the respective graphs
    return fig_bar, fig_line

if __name__ == '__main__':
    #app.run(debug=True)
    app.run_server(debug=True)
    #app.run(debug=True)
