import pandas as pd
import numpy as np
import pickle

with open('pickle/recommendation.pkl', 'rb') as data:
    userRecommendation = pickle.load(data)

with open('pickle/model.pkl', 'rb') as model:
    best_model = pickle.load(model)

with open('pickle/tfidf_vectorizer.pkl', 'rb') as vectorizer:
    tfidf = pickle.load(vectorizer)

def get_sentiment(text):
    sentiment_id = best_model.predict(tfidf.transform([text]))
    return sentiment_id

def predict_top5(username):
    top_20 = userRecommendation.loc[str(username)].sort_values(ascending=False)[0:20]
    ratings = pd.read_csv('data/ratings_df.csv' , encoding='latin-1')
    output_user = pd.merge(top_20,ratings,left_on='name',right_on='name',how='left')
    rows = []
    for text,name in zip(output_user['reviews_text'],output_user['name']):
        rows.append([name,text,get_sentiment(text)])
    
    df = pd.DataFrame(rows, columns=["product","review", "sentiment"])
    output = pd.DataFrame(((df.groupby('product')['sentiment'].sum() / df.groupby('product')['review'].count())*100).sort_values(ascending=False)[0:5])
    output = output.reset_index()
    output_list = output.values.tolist()

    return output_list