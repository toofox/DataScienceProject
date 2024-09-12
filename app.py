import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from layout import homepage, projects_section, about_section
from layout import (df_combined_Uni_FH, df_combined_Uni_Germany, df_combined_Uni_EU,
                    df_combined_Uni_Asia, df_combined_Uni_World, df_combined_FH_Germany,
                    df_combined_FH_EU, df_combined_FH_Asia, df_combined_FH_World,
                    df_combined_Germany_EU, df_combined_Germany_Asia, df_combined_Germany_World,
                    df_combined_EU_Asia, df_combined_EU_World, df_combined_Asia_World,
                    task_workflow)

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
    [Output('comparison-bar-chart7', 'figure'),
     Output('comparison-line-chart7', 'figure')],
    Input('dataset-combination-dropdown', 'value')
)
def update_combination_graph(selected_combination):
    # Access the correct dataset combination dynamically
    combined_df = globals()[selected_combination]

    # Generate both the bar and line charts using task_workflow
    fig_bar, fig_line = task_workflow(combined_df, [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], "Selected Combination")

    # Return both figures to update the respective graphs
    return fig_bar, fig_line

if __name__ == '__main__':
    #app.run(debug=True)
    app.run_server(debug=True)
    #app.run(debug=True)
