from dash import Dash, dcc, Output, Input, State, html, ctx  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import plotly.express as px
import pandas as pd
import utility
import geo

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# epidemic tab components
epi_time = utility.get_available_pulled_time()
epi_start_dropdown = dcc.Dropdown(id='start-time-dropdown',
                                  options=epi_time,
                                  value=epi_time[0],
                                  clearable=False,
                                  style={'width': '150px'})

epi_end_dropdown = dcc.Dropdown(id='end-time-dropdown',
                                options=epi_time,
                                value=epi_time[1],
                                clearable=False,
                                style={'width': '150px'})

epi_button = html.Button('Analyse', id='epi-analyse-button', n_clicks=0, style={'width': '150px', 'position': 'relative', 'bottom': '18px'})
epi_hist_graph = dcc.Graph(id='sentiment-hist-graph', figure={})
epi_pos_graph = dcc.Graph(id='pos-word-graph', figure={})
epi_neu_graph = dcc.Graph(id='neu-word-graph', figure={})
epi_neg_graph = dcc.Graph(id='neg-word-graph', figure={})

# geo tab components
geo_graph = dcc.Graph(id='geo-graph', figure={})

# vaccine tab
vaccine_graph = dcc.Graph(id='vaccine-graph', figure={})

# predict tab components
predict_button = html.Button('Analyse', id='predict-analyse-button', n_clicks=0, style={'width': '150px', 'position': 'relative', 'bottom': '18px'})
predict_sub_graph = dcc.Graph(id='predict-sub-gauge', figure={}, style={
                                    "display": "inline-block",
                                })
predict_pos_graph = dcc.Graph(id='predict-pos-gauge', figure={})
predict_neg_graph = dcc.Graph(id='predict-neg-gauge', figure={})

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H1('COVID-19 Sentiment Analyser', style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs", value='epidemic-tab', children=[
        dcc.Tab(label='Epidemic', value='epidemic-tab', children=[
            html.Div([
                html.H3('Choose a time period', style={'textAlign': 'center'}),
                html.Div([
                    epi_start_dropdown,
                    html.Div([
                        html.H3('-', style={'vertical-align': 'top', 'position': 'relative', 'bottom': '18px'})
                    ]),
                    epi_end_dropdown,
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%', 'margin': '0 auto'}),
                html.Div([
                    epi_button
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%', 'margin': '0 auto'}),
                dcc.Loading(id = "loading-icon1", children=[epi_hist_graph, epi_pos_graph, epi_neu_graph, epi_neg_graph], type="default"),
                # epi_hist_graph,
                # epi_pos_graph,
                # epi_neu_graph,
                # epi_neg_graph
            ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'padding-left': '20px', 'padding-right': '20px'})
        ]),
        dcc.Tab(label='Geographic', value='geo-tab', children=[
            html.Div([
                html.H3('Sentiments based on countries', style={'textAlign': 'center'}),
                # html.Label(['Sentiments based on countries']),
                html.Div(
                    dcc.Dropdown(
                        geo.countryDict, 'United States', 
                        multi=False, id='selected_country',
                        clearable = False,
                        style = {'width': '200px', 'textAlign': 'center', 'position': 'relative', 'justify-content': 'center'},
                    ),
                    style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%', 'margin': '0 auto'}),
                dcc.Loading(id = "loading-icon2", children=[html.Div(dcc.Graph(id='geo-graph'))], type="default"),
            ])
        ]),
        dcc.Tab(label='Vaccine', value='vaccine-tab', children=[
            html.Div([
                html.H3('Sentiment of Tweets related to vaccines', style={'textAlign': 'center'}),
                # html.Label(['Sentiments related to vaccines']),
                html.Div(dcc.Dropdown(
                    options = [
                        {'label': 'All Vaccines', 'value': 'vaccine'},
                        {'label': 'Pfizer', 'value': 'Pfizer'},
                        {'label': 'Moderna','value': 'Moderna'},
                        {'label': 'Covaxin','value': 'Covaxin'},
                        {'label': 'AstraZeneca','value': 'AstraZeneca'},
                        {'label': 'Covishield','value': 'Covishield'},
                        {'label': 'Janssen', 'value': 'Janssen'},
                        {'label': 'Sinovac', 'value': 'Sinovac'},
                        {'label': 'Sinopharm', 'value': 'Sinopharm'},
                    ],
                    value = 'vaccine', id='selected_keyword',
                    clearable = False,
                    style = {'width': '200px', 'textAlign': 'center', 'position': 'relative', 'justify-content': 'center'},
                ), style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%', 'margin': '0 auto'}),
            ]),
            dcc.Loading(id = "loading-icon3", 
                children=[html.Div(dcc.Graph(id='vaccine-graph'))], type="default")
        ]),
        dcc.Tab(label='Predict', value='predict-tab', children=[
            html.Div([
                html.H3('Input tweet you want to analyse below.', style={'textAlign': 'center'}),
                html.Div([dcc.Textarea(id='input-to-predict', style={'width': '350px', 'height':'150px', 'vertical-align': 'top', 'text-align': 'left'})], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%', 'height': '200px','margin': '0 auto'}),
                html.Div([
                    predict_button
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%', 'margin': '0 auto'}),
                html.H3(id='predict_result', children='Sentiment Result: ', style={'textAlign': 'center'}),
                html.Div([
                    predict_sub_graph,
                    predict_pos_graph,
                    predict_neg_graph
                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%', 'margin': '0 auto'}),
            ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'padding-left': '20px', 'padding-right': '20px', 'margin': '0 auto', 'text-align': 'center'})
        ]),
    ], style={'padding-left': '20px', 'padding-right': '20px', 'font-size': 'large', 'font-weight': 'bold'}),
])


@app.callback(
    Output(epi_hist_graph, component_property='figure'),
    Output(epi_pos_graph, component_property='figure'),
    Output(epi_neu_graph, component_property='figure'),
    Output(epi_neg_graph, component_property='figure'),
    Input(epi_button, component_property='n_clicks'),
    State(epi_start_dropdown, component_property='value'),
    State(epi_end_dropdown, component_property='value')
)
def update_epidemic(n_clicks, start_time, end_time):
    if 'epi-analyse-button' == ctx.triggered_id:
        result = utility.generate_epidemic_graphs(start_time, end_time)
        if result is None:
            print('no result')
            return {}, {}, {}, {}
        return result
    return {}, {}, {}, {}


@app.callback(
    Output(geo_graph, component_property='figure'),
    Input("selected_country", "value"), 
)
def update_graph(selected_country):
    df = utility.get_geo_data(selected_country)
    fig = px.pie(df, names = 'sentiment', hole=.3)
    return fig

@app.callback(
    Output(vaccine_graph, component_property='figure'),
    Input("selected_keyword", "value"), 
)
def update_graph(selected_keyword):
    df = utility.get_vaccine_data(selected_keyword)
    fig = px.pie(df, names = 'sentiment', hole=.3,)
                # labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
    return fig

@app.callback(
    Output(predict_sub_graph, component_property='figure'),
    Output(predict_pos_graph, component_property='figure'),
    Output(predict_neg_graph, component_property='figure'),
    Output('predict_result', 'children'),
    Input(predict_button, component_property='n_clicks'),
    State('input-to-predict', 'value')
)
def update_epidemic(n_clicks, text_to_predict):
    if 'predict-analyse-button' == ctx.triggered_id:
        result = utility.generate_predict_graphs(text_to_predict)
        if result is None:
            return {}, {}, {}, 'Sentiment Result: '
        return result
    return {}, {}, {}, 'Sentiment Result: '


# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8054)
