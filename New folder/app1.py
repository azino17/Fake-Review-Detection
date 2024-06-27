from flask import Flask, render_template, request
import pandas as pd
from google_play_scraper import app, Sort, reviews
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

nltk.download('vader_lexicon')

app = Flask(__name__)
sid = SentimentIntensityAnalyzer()

picFolder = os.path.join('static', 'pics')
if not os.path.exists(picFolder):
    os.makedirs(picFolder)  # Create the directory if it doesn't exist
app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.jpg')
    return render_template('NEXT.html', user_image=pic1)

def classify_review(sentiment_score):
    """
    Classify review as genuine or fake based on sentiment score.
    """
    if sentiment_score >= 0.8 or sentiment_score <= -0.8:
        return 'FAKE'
    else:
        return 'GENUINE'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form.get('app_id')
        print("App ID:", input_text)
        app_packages = [input_text]
        result, _ = reviews(
            app_packages[0],
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=1500
        )
        print("Number of reviews retrieved:", len(result))
        df = pd.json_normalize(result)
        df['content'] = df['content'].astype('str')
        df['sentiment'] = df['content'].apply(lambda x: sid.polarity_scores(x)['compound'])
        df['classification'] = df['sentiment'].apply(classify_review)
        fig = px.histogram(df, x='classification', color='classification', text_auto=True)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        fig.write_image(os.path.join(app.config['UPLOAD_FOLDER'], 'result_plot.png'))
        genuine_count = df['classification'].value_counts().get('GENUINE', 0)
        fake_count = df['classification'].value_counts().get('FAKE', 0)

        if genuine_count > fake_count:
            recommendation = "The majority of reviews are genuine."
        else:
            recommendation = "A significant number of reviews are suspected to be fake."

        avg_rating = round(df['score'].mean(), 1)
        print("Recommendation:", recommendation)
        print("Average Rating:", avg_rating)

        pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'result_plot.png')
        return render_template('result.html', recommendation=recommendation, rating=avg_rating, user_image=pic2)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
