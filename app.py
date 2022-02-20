from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

with open('pickle/User_Recommendation.pkl', 'rb') as data:
    userRecommendation = pickle.load(data)

with open('pickle/Logistic_Regression.pkl', 'rb') as model:
    best_model = pickle.load(model)

with open('pickle/Vectorizer.pkl', 'rb') as vectorizer:
    tfidf = pickle.load(vectorizer)

def get_sentiment(text):
    """
    Predicts the sentiment of text using the Multinomial Naive Bayes Model
    """
    sentiment_id = best_model.predict(tfidf.transform([text]))
    return sentiment_id

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    username = request.form.get('username')
    print(username)
    d = userRecommendation.loc[str(username)].sort_values(ascending=False)[0:20]
    ratings = pd.read_csv('data/ratings_df.csv' , encoding='latin-1')
    output_user = pd.merge(d,ratings,left_on='name',right_on='name',how='left')
    rows = []
    for text,name in zip(output_user['reviews_text'],output_user['name']):
        rows.append([name,text,get_sentiment(text)])
    
    df = pd.DataFrame(rows, columns=["product","review", "sentiment"])

    return render_template("index.html", OUTPUT=((df.groupby('product')['sentiment'].sum() / df.groupby('product')['review'].count())*100).sort_values(ascending=False)[0:5])

if __name__ == '__main__':
    app.run(debug=True)
