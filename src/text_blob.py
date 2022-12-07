import nltk

#### run this once ###
# nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import twitter


def predict_sentiment(text):
    tweet_dict = {"text": text}
    tweet_list_df = pd.DataFrame(tweet_dict)
    cleaned_tweets = []
    for tweet in tweet_list_df['text']:
        cleaned_tweet = twitter.preprocess_tweet(tweet)
        cleaned_tweets.append(cleaned_tweet)
    tweet_list_df['cleaned_text'] = pd.DataFrame(cleaned_tweets)
    tweet_list_df[['polarity', 'subjectivity']] = tweet_list_df['cleaned_text'].apply(lambda t: pd.Series(TextBlob(t).sentiment))
    for index, row in tweet_list_df['cleaned_text'].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if comp <= -0.05:
            tweet_list_df.loc[index, 'sentiment'] = "negative"
        elif comp >= 0.05:
            tweet_list_df.loc[index, 'sentiment'] = "positive"
        else:
            tweet_list_df.loc[index, 'sentiment'] = "neutral"
        tweet_list_df.loc[index, 'neg'] = neg
        tweet_list_df.loc[index, 'neu'] = neu
        tweet_list_df.loc[index, 'pos'] = pos
        tweet_list_df.loc[index, 'compound'] = comp
    return tweet_list_df


if __name__ == '__main__':
    result = predict_sentiment(["I believe Trump is the worst president and he should be responsible for all those who died from the corona virus."])
    print(result['subjectivity'][0])
    print(result['neg'][0])
    print(result['pos'][0])