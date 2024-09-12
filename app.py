import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
from layout import homepage, projects_section, about_section, task_workflow
import plotly.express as px

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

###### IMPORT SECTION ######

RQ_4_df = pd.read_csv('Data/RQ4+6/RQ4_PDF_Abstract.csv')
RQ_6_df = pd.read_csv('Data/RQ4+6/RQ6_faculty_keyword.csv')

############################

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

###### CALLBACK SECTION ######

# RQ4 callback function

@app.callback(Output(component_id='RQ4-barchart', component_property='figure'),
              [Input(component_id='RQ4-year-slider', component_property='value'),
               Input(component_id='RQ4-Keyword-radio', component_property='value'),
               Input(component_id='RQ4-Keyword-radio-log', component_property='value'),])
def graph_rq4_update(slider_value, radio_value, xaxis_type):

    filtered_df = RQ_4_df[RQ_4_df['Year'] == slider_value]

    if radio_value == 'Count_PDF':
        x_value = filtered_df['Count_PDF']
    elif radio_value == 'Count_Abs':
        x_value = filtered_df['Count_Abs']
    else:
        x_value = filtered_df['Count_PDF']

    fig = px.bar(filtered_df,
                 x=x_value,
                 y=filtered_df['Keyword'],
                 color="Keyword", hover_name="Keyword",
                 orientation='h',
                 log_x=xaxis_type,)

    fig.update_layout(
        height=800,
        width=800,
        xaxis_title=radio_value,
        yaxis=dict(showgrid=True, showline=True, ticks='outside', tickson='boundaries', automargin=True),
        xaxis=dict(showgrid=True, showline=True, ticks='outside', tickson='boundaries', automargin=True)
    )

    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'})

    fig.update_xaxes(rangemode="tozero")

    return fig

# RQ6 Callback
@app.callback(Output(component_id='RQ6-scatter', component_property='figure'),
              [Input(component_id='RQ6-year-slider', component_property='value'),
               Input(component_id='RQ6-Keyword-dropdown', component_property='value')])
def graph_update(slider_value, keyword_value):
    all_faculties = RQ_6_df['Faculty'].unique()
    all_keywords = RQ_6_df['Keyword'].unique()

    # depending on dropdown selection display legend
    if keyword_value == 'All':
        filtered_df = RQ_6_df.loc[RQ_6_df['Year'] == slider_value]
    else:
        filtered_df = RQ_6_df.loc[(RQ_6_df['Year'] == slider_value) & (RQ_6_df['Keyword'] == keyword_value)]

    grouped_df = filtered_df.groupby(['Faculty', 'Keyword'], as_index=False)['Count'].sum()

    # Need to make a missing_faculties list to always have all faculties displayed
    missing_faculties = set(all_faculties) - set(grouped_df['Faculty'])
    if missing_faculties:
        missing_data = pd.DataFrame({
            'Keyword': [keyword_value] * len(missing_faculties),
            'Faculty': list(missing_faculties),
            'Count': [0] * len(missing_faculties),
            'Year': [slider_value] * len(missing_faculties)
        })
    grouped_df = pd.concat([grouped_df, missing_data])

    fig = px.scatter(grouped_df, x="Keyword", y="Faculty",
                     size="Count", color="Keyword", hover_name="Keyword",
                     size_max=55,
                     category_orders={"Faculty": all_faculties, "Keyword": all_keywords},
                     )

    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': all_keywords},  # Fix x-axis to show all keywords
        yaxis={'categoryorder': 'array', 'categoryarray': all_faculties},  # Fix y-axis to show all faculties
        xaxis_title='Keyword',
        yaxis_title='Faculty'
    )
    return fig

##############################


if __name__ == '__main__':
    #app.run(debug=True)
    app.run_server(debug=True)
    #app.run(debug=True)
