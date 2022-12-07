import csv
import pandas as pd

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import os

# you need to include __name__ in your Dash constructor if
# you plan to use a custom CSS or JavaScript in your Dash apps

#app = dash.Dash(__name__)

frame = {'tweet_id':[], 'country':[], 'sentiment':[]}
countryDict = []
country_senti = []
flag = 1
dff=None

# __location__ = os.path.realpath(
#     os.path.join(os.getcwd(), os.path.dirname(__file__)))
# geopath = os.path.join(__location__, 'geodata.pkl')

geopath = 'geodata.pkl'

if flag == 1:
    dff = pd.read_pickle(geopath)
    countryDict = dff.country.unique()
else:
    with open(r'C:\Users\xiaok\Downloads\vaccine_tweetid_userid_keyword_sentiments_emotions.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        i = 1
        for row in reader:
            # country : sentiment
            if not row['country/region'] == '-':
                loc = row['country/region']
                country_senti.append([loc, row['sentiment']])
                #country_senti.setdefault(loc,[]).append(row['sentiment'])
                #frame['tweet_id'].append(row['tweet_ID'])
                #frame['country'].append(row['country/region'])
                #frame['sentiment'].append(row['sentiment'])
                if row['country/region'] not in countryDict:
                    countryDict.append(row['country/region'])
                i += 1
                if i % 50000 == 0:
                    print('====50k===')
        dff = pd.DataFrame(country_senti, columns = ['country', 'sentiment'])
        print("saving data...")
        dff.to_pickle('geodata.pkl')

"""
#---------------------------------------------------------------
app.layout = html.Div([
    html.Div([
        #Html setup and label
        html.Label(['Sentiments based on countries']),
        dcc.Dropdown(countryDict, 'United States', multi=False, id='selected_country'),
        dcc.Graph(id='the_graph')
    ]),
])

#---------------------------------------------------------------
@app.callback(
    Output("the_graph", "figure"), 
    Input("selected_country", "value"), 
)

def update_graph(selected_country):
    df = dff.loc[dff['country'] == selected_country]
    fig = px.pie(df, names = 'sentiment', hole=.3)
    print("loaded?")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
"""
