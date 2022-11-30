import os
import json
import re
import tweepy
import numpy as np
import unicodedata
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


def get_pulled_tweet_by_date(date):
    file_name = '../raw_data/' + date + '_short.json'
    try:
        data_df = pd.read_json(file_name, lines=True)
    except ValueError:
        return None
    data_df['cleaned_text'] = data_df['text'].apply(preprocess_tweet)
    return data_df


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
            curr_df = get_pulled_tweet_by_date(date)
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



if __name__ == '__main__':
    #print(get_pulled_tweet_by_date('2020-26'))
    manual_label_tweets(2020, 10, 12)
