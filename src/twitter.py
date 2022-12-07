import os
import re
import tweepy
import numpy as np
import log_regression
import utility
import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
bearer_token = os.getenv('BEARER_TOKEN')
client = tweepy.Client(bearer_token=bearer_token)


def preprocess_tweet(sen):
    """Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase"""
    sentence = sen.lower()
    # Remove RT
    sentence = re.sub('RT @\w+: ', " ", sentence)
    # Remove Tag
    sentence = re.sub('(#[A-Za-z0-9]+)', '', sentence)
    # Remove special characters
    sentence = re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def get_tweet_by_id(input_id):
    tweet = client.get_tweet(id=input_id)
    if tweet.data is None:
        return None
    return str(tweet.data)


def search_covid_tweet_by_date(date, max_tweets=50):
    """search tweets at most one week before """
    today = datetime.date.today()
    d1 = today.strftime("%d/%m/%Y")
    _, m, y = d1.split('/')
    query = '#COVID19 -is:retweet lang:en'
    since_date = datetime.datetime(year=int(y), month=int(m), day=date, hour=0, minute=0, second=0, microsecond=0,
                                   tzinfo=datetime.timezone.utc)
    until_date = datetime.datetime(year=int(y), month=int(m), day=date + 1, hour=0, minute=0, second=0, microsecond=0,
                                   tzinfo=datetime.timezone.utc)
    try:
        tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at', 'geo'],
                                             max_results=max_tweets, start_time=since_date.isoformat(),
                                             end_time=until_date.isoformat())
        return tweets.data
    except BaseException:
        return None


def get_pulled_tweet_by_date(date):
    """ get tweets from raw_data """
    file_name = '../raw_data/' + date + '_short.json'
    try:
        data_df = pd.read_json(file_name, lines=True)
        print('success')
    except ValueError:
        print('cannot read data')
        return None
    return data_df


def get_pulled_tweet_by_time_period(start_time, end_time):
    start_year, start_month = start_time.split('-')
    end_year, end_month = end_time.split('-')
    start = int(start_year) * 12 + int(start_month)
    end = int(end_year) * 12 + int(end_month)
    if start > end:
        return None
    sample_count = 1000
    if (end - start) > 3:
        sample_count = 100
    file_name = '../raw_data/' + start_time + '-01_short.json'
    data_df = pd.read_json(file_name, lines=True)
    data_df = data_df.drop(labels=range(1, len(data_df.index)), axis=0)
    for i in range(start, end + 1):
        month = i % 12
        year = int(i / 12)
        if month == 0:
            month = 12
            year -= 1
        year_s = str(year) + '-'
        month_s = str(month) + '-'
        if len(month_s) == 2:
            month_s = '0' + month_s
        for day in range(1, 32):
            day_s = str(day)
            if len(day_s) == 1:
                day_s = '0' + day_s
            print('get ' + year_s + month_s + day_s + ' data')
            df = get_pulled_tweet_by_date(year_s + month_s + day_s)
            if df is None:
                continue
            no_retweet_df = df[df['is_retweet'] == False]
            random_idx = np.random.choice(range(len(no_retweet_df.index)), sample_count, replace=False)
            data_df = data_df.append(no_retweet_df.iloc[random_idx], ignore_index=True)
    print('preprocess all tweets\n')
    data_df['cleaned_text'] = data_df['text'].apply(preprocess_tweet)
    print('text preprocessing done\n')
    print('totally ' + str(len(data_df.index) - 1) + ' tweets retrieved.')
    return data_df


if __name__ == '__main__':
    result = search_covid_tweet_by_date(25, 10)
    # print(result[0].text)
    # result_df = get_pulled_tweet_by_time_period('2021-01', '2021-01')
    # prediction = log_regression.load_model_to_predict(input_text=result_df['cleaned_text'])
    # result_df['pred'] = prediction
    # result_df.loc[result_df["pred"] == 1, "sentiment"] = 'positive'
    # result_df.loc[result_df["pred"] == 0, "sentiment"] = 'neutral'
    # result_df.loc[result_df["pred"] == -1, "sentiment"] = 'negative'
    # fig = utility.generate_sentiment_hist(result_df, 'sentiment')
