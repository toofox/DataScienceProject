import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
from layout import homepage, projects_section, about_section
import plotly.express as px

# Initialize Dash app with external Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

###### IMPORT SECTION ######

RQ_4_df = pd.read_csv('Data/RQ4_PDF_Abstract.csv')
RQ_6_df = pd.read_csv('Data/RQ6_faculty_keyword.csv')

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
    app.run(debug=True)
    #app.run_server(debug=True)
    #app.run(debug=True)
