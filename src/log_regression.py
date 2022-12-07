import json
import numpy as np
import pandas as pd
import utility
import twitter
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


def train_model(input_file_name='../labeled_data/gold.json', train_perc=0.8, max_iter=10000, solver='newton-cg', model_name="../model/log_model.sav", vec_name="../model/log_train.json"):
    sem_df = pd.read_json(input_file_name, lines=True)
    sem_df.loc[sem_df["label"] == "positive", "sentiment"] = 1
    sem_df.loc[sem_df["label"] == "neutral", "sentiment"] = 0
    sem_df.loc[sem_df["label"] == "negative", "sentiment"] = -1

    df = sem_df[['cleaned_text', 'sentiment']].copy()
    index = df.index
    df['random_number'] = np.random.randn(len(index))
    train = df[df['random_number'] <= train_perc]
    test = df[df['random_number'] > train_perc]

    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train['cleaned_text'])
    test_matrix = vectorizer.transform(test['cleaned_text'])
    x_train = train_matrix
    x_test = test_matrix
    y_train = train['sentiment']
    y_test = test['sentiment']

    lr = LogisticRegression(solver=solver, max_iter=max_iter, class_weight='balanced', tol=1e-6)
    lr.fit(x_train, y_train)
    joblib.dump(lr, model_name)
    train.to_json(vec_name, orient="records", lines=True)

    predictions = lr.predict(x_test)
    print(classification_report(predictions, y_test))
    return sum(predictions == y_test) / len(y_test)


def load_model_to_predict(input_text, model_file_name="../model/log_model.sav", train_vec_name="../model/log_train.json"):
    lr = joblib.load(model_file_name)
    tweets = {"cleaned_text":[]}
    for t in input_text:
        tweets['cleaned_text'].append(twitter.preprocess_tweet(t))
    df = pd.DataFrame(tweets)
    train_df = pd.read_json(train_vec_name, lines=True)
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    _ = vectorizer.fit_transform(train_df['cleaned_text'])
    matrix = vectorizer.transform(df['cleaned_text'])
    predictions = lr.predict(matrix)
    return predictions


if __name__ == '__main__':
    # acc = train_model()
    # sem_df = pd.read_json('../labeled_data/gold.json', lines=True)
    # sem_df.loc[sem_df["label"] == "positive", "sentiment"] = 1
    # sem_df.loc[sem_df["label"] == "neutral", "sentiment"] = 0
    # sem_df.loc[sem_df["label"] == "negative", "sentiment"] = -1
    # pred = load_model_to_predict(input_text=sem_df['cleaned_text'])
    # ans = sem_df['sentiment']
    # print(sum(pred == ans) / len(ans))

    # pred = load_model_to_predict(input_text=['Hope...it plays a great role...', "The death of Perence Shiri is a direct consequence of his government's  lack of preparedness.", "Russia is set to become the first country to approve a coronavirus vaccine"])
    pred = load_model_to_predict(input_text=[
        "Health authorities in Australia's Victoria state ramped up contact tracing and prepared for more mass testing of re\u2026 https:\/\/t.co\/ohhe8oSDz4",
        "Great to see a good news story about a dad during #COVID19. https:\/\/t.co\/PLAqxKFvDO",
        "Romance fraud on rise in coronavirus lockdown https:\/\/t.co\/NKiHU4YN2M via @BBCNews Now whats your plan\ud83d\ude02"])
    print(pred)
