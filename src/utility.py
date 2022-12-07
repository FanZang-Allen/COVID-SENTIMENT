import numpy as np
import log_regression
import pandas as pd
import twitter
import text_blob
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from plotly.subplots import make_subplots

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
### run this once ###
# nltk.download('stopwords') # only once

frame = {'tweet_id':[], 'country':[], 'sentiment':[]}
countryDict = []
country_senti = []
flag = 1
dff = None

geopath = 'geodata.pkl'
# vaccine_path = os.path.join(__location__, 'vaccine_sentiment.pkl')
vaccine_path = 'vaccine_sentiment.pkl'
vaccine_df = pd.read_pickle(vaccine_path)

if flag == 1:
    # dff = pd.read_pickle('./geodata.pkl')
    dff = pd.read_pickle(geopath)
    countryDict = dff.country.unique()


def get_geo_data(selected_country):
    df = dff.loc[dff['country'] == selected_country]
    return df

def get_vaccine_data(selected_keyword):
    df = vaccine_df.loc[vaccine_df['keywords'] == selected_keyword]
    return df

def generate_word_cloud(connected_str, img_name="../visualization_data/word_cloud.png", title="Word Cloud"):
    stopwords1 = set(stopwords.words('english'))
    stopwords1.update(["br", "href"])
    wl = WordCloud(stopwords=stopwords1).generate(connected_str)
    # fig = make_subplots(
    #     rows=1, cols=2,
    #     subplot_titles=("Plot 1", "Plot 2"))
    # fig.add_trace(px.imshow(wl).data[0], row = 1, col = 1)
    # fig.add_trace(px.imshow(wl).data[0], row = 1, col = 2)
    fig = px.imshow(wl)
    fig.update_layout(title_text=title, title_x=0.5, font=dict(
        family="Courier New, monospace",
        size=22,
        color="Black"
    ))
    fig.write_image(img_name)
    return fig


def generate_sentiment_hist(start_time, end_time, title="Sentiment Histogram", img_name="../visualization_data/hist.png"):
# def generate_sentiment_hist(start_time, end_time, title="Sentiment Histogram", img_name="hist.png"):
    """
    run pip install -U kaleido first
    """
    result_df = twitter.get_pulled_tweet_by_time_period(start_time, end_time)
    if result_df is None:
        print('generate_sentiment_hist: no result_df')
        return None, None
    print('Start sentiment analysis.\n')
    prediction = log_regression.load_model_to_predict(input_text=result_df['cleaned_text'])
    print('Sentiment analysis finished. \n')
    result_df['pred'] = prediction
    result_df.loc[result_df["pred"] == 1, "sentiment"] = 'positive'
    result_df.loc[result_df["pred"] == 0, "sentiment"] = 'neutral'
    result_df.loc[result_df["pred"] == -1, "sentiment"] = 'negative'

    fig = px.histogram(result_df, x="sentiment")
    fig.update_traces(marker_color="indianred", marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5)
    fig.update_layout(title_text=title, title_x=0.5)
    print('1')
    fig.write_image(img_name)
    print('2')
    return fig, result_df


def generate_epidemic_graphs(start_time, end_time):
    hist_fig, data_df = generate_sentiment_hist(start_time, end_time)
    if hist_fig is None:
        print('generate_epidemic_graphs: no hist fig')
        return None
    print('Sentiment Histogram Finished\n')
    positive = data_df[data_df['pred'] == 1]
    neutral = data_df[data_df['pred'] == 0]
    negative = data_df[data_df['pred'] == -1]
    pos_str = " ".join(t for t in positive['cleaned_text'])
    neu_str = " ".join(t for t in neutral['cleaned_text'])
    neg_str = " ".join(t for t in negative['cleaned_text'])
    pos_word_fig = generate_word_cloud(pos_str, img_name="../visualization_data/positive_word_cloud.png",
                                       title="Positive Word Cloud")
    print('Positive Word Cloud Finished\n')
    neu_word_fig = generate_word_cloud(neu_str, img_name="../visualization_data/neutral_word_cloud.png",
                                       title="Neutral Word Cloud")
    print('Neutral Word Cloud Finished\n')
    neg_word_fig = generate_word_cloud(neg_str, img_name="../visualization_data/negative_word_cloud.png",
                                       title="Negative Word Cloud")
    print('Negative Word Cloud Finished\n')
    return hist_fig, pos_word_fig, neu_word_fig, neg_word_fig


def manual_label_tweets(year, start_month, end_month):
    file_name = '../labeled_data/gold.json'
    data_df = pd.read_json(file_name, lines=True)
    ids = set(data_df['id_str'])
    for month in range(start_month, end_month + 1):
        for day in range(1, 29):
            m_str = str(month)
            if len(m_str) < 2:
                m_str = '0' + m_str
            d_str = str(day)
            if len(d_str) < 2:
                d_str = '0' + d_str
            date = str(year) + '-' + m_str + '-' + d_str
            curr_df = twitter.get_pulled_tweet_by_date(date)
            curr_df['cleaned_text'] = curr_df['text'].apply(twitter.preprocess_tweet)
            if curr_df is None:
                continue
            no_retweet_df = curr_df[curr_df['is_retweet'] == False]
            random_idx = np.random.choice(range(len(no_retweet_df.index)), 10, replace=False)
            unlabeled_df = no_retweet_df.iloc[random_idx].copy()
            unlabeled_df['label'] = "positive"
            while True:
                print('start to label tweets for ' + date + '\n')
                unlabeled_df['label'] = "positive"
                for i in unlabeled_df.index:
                    if unlabeled_df["id_str"][i] in ids:
                        continue
                    print('Original Tweet:\n')
                    print(unlabeled_df["text"][i] + '\n')
                    print('Processed Tweet:\n')
                    print(unlabeled_df["cleaned_text"][i] + '\n')
                    while True:
                        sen = input('Please type the sentiment: 1 for positive, 0 for neutral, -1 for negative \n')
                        if int(sen) == 1:
                            unlabeled_df.loc[i, "label"] = "positive"
                            break
                        elif int(sen) == 0:
                            unlabeled_df.loc[i, "label"] = "neutral"
                            break
                        elif int(sen) == -1:
                            unlabeled_df.loc[i, "label"] = "negative"
                            break
                        else:
                            continue
                c = input("Satisfied with labels for this date? y/n \n")
                if c == 'y':
                    for i in unlabeled_df.index:
                        if unlabeled_df["id_str"][i] not in ids:
                            data_df = data_df.append(unlabeled_df.loc[i], ignore_index = True)
                            ids.add(unlabeled_df["id_str"][i])
                    break
            data_df.to_json(file_name, orient="records", lines=True)


def get_available_pulled_time():
    result = []
    for i in range(8, 13):
        text = '2020-'
        if i < 10:
            text += '0' + str(i)
        else:
            text += str(i)
        result.append(text)
    for i in range(1, 8):
        text = '2021-'
        if i < 10:
            text += '0' + str(i)
        else:
            text += str(i)
        result.append(text)
    return result


def generate_gauge_char(min_bound, max_bound, pred_value, is_sentiment, title, img_name="../visualization_data/gauge_chart.png"):
    plot_bgcolor = "white"
    quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229"]
    quadrant_text = ["", "<b>High</b>", "<b>Medium</b>", "<b>Low</b>"]
    if is_sentiment is False:
        quadrant_text = ["", "<b>Mostly Subjective</b>", "<b>Subjective & Objective</b>", "<b>Mostly Objective</b>"]
    n_quadrants = len(quadrant_colors) - 1

    current_value = pred_value
    min_value = min_bound
    max_value = max_bound
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, t=10, l=10, r=10),
            width=450,
            height=400,
            paper_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    text="<b>" + title+"</b>" + f": {current_value}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.35, yanchor="bottom", yref="paper",
                    showarrow=False,
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    fig.write_image(img_name)
    return fig


def generate_predict_graphs(text):
    df = text_blob.predict_sentiment([text])
    sub_score = df['subjectivity'][0]
    neg_score = df['neg'][0]
    pos_score = df['pos'][0]
    final_sentiment = df['sentiment'][0]
    print('sentiment analyse finished \n')
    sub_fig = generate_gauge_char(0, 1, sub_score, False, title='Subjectivity Score', img_name="../visualization_data/sub_gauge_chart.png")
    print('subjectivity graph finished \n')
    pos_fig = generate_gauge_char(0, 1, pos_score, True, title='Positive Score', img_name="../visualization_data/pos_gauge_chart.png")
    print('positive graph finished \n')
    neg_fig = generate_gauge_char(0, 1, neg_score, True, title='Negative Score', img_name="../visualization_data/neg_gauge_chart.png")
    print('negative graph finished \n')
    return sub_fig, pos_fig, neg_fig, 'Sentiment Result: ' + final_sentiment


if __name__ == '__main__':
    # generate_word_cloud("sad i want to show this this this this this")
    # print(get_available_pulled_time())
    # generate_epidemic_graphs('2021-01', '2021-01')
    generate_gauge_char(0, 1, 0.33, False, title='Subjectivity Score',
                        img_name="../visualization_data/gauge_chart.png")