from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    username = request.form.get('username')
    finalRecommendation = model.predict_top5(username)
    output_df = pd.DataFrame(finalRecommendation, columns=["product","percentage"])
    output = output_df['product'].tolist()
    return render_template("index.html", OUTPUT= output , Text = 'Recommended Products')

if __name__ == '__main__':
    app.run(debug=True)
