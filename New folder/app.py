from flask import Flask, render_template, request
import pandas as pd
from google_play_scraper import Sort, reviews
import plotly.express as px
import joblib
import os
import re
import nltk

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

app = Flask(__name__)

picFolder = os.path.join('static', 'pics')
if not os.path.exists(picFolder):
    os.makedirs(picFolder)
app.config['UPLOAD_FOLDER'] = picFolder

# Load the pre-trained model and TF-IDF vectorizer
rf_model = joblib.load('random_forest_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.jpg')
    return render_template('NEXT.html', user_image=pic1)

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

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

        # Ensure no empty reviews
        df = df[df['content'].str.strip() != '']
        print(f"Number of non-empty reviews: {len(df)}")

        # Preprocess reviews
        df['content'] = df['content'].apply(preprocess_text)

        # Ensure no empty reviews after preprocessing
        df = df[df['content'].str.strip() != '']
        print(f"Number of non-empty reviews after preprocessing: {len(df)}")

        # Transform reviews to numerical features
        X_reviews = tfidf_vectorizer.transform(df['content']).toarray()

        # Predict using the pre-trained model
        df['classification'] = rf_model.predict(X_reviews)
        df['classification'] = df['classification'].apply(lambda x: 'Human' if x == 0 else 'Computer')

        # Log the counts of each classification
        print("Classification counts:", df['classification'].value_counts())

        # Generate histogram
        fig = px.histogram(df, x='classification', color='classification', text_auto=True)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        fig.write_image(os.path.join(app.config['UPLOAD_FOLDER'], 'result_plot.png'))

        # Provide recommendation based on classification counts
        human_count = df['classification'].value_counts().get('Human', 0)
        computer_count = df['classification'].value_counts().get('Computer', 0)

        if human_count > computer_count:
            recommendation = "The majority of reviews are human-generated."
        else:
            recommendation = "A significant number of reviews are computer-generated."

        avg_rating = round(df['score'].mean(), 1)
        print("Recommendation:", recommendation)
        print("Average Rating:", avg_rating)

        pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'result_plot.png')
        return render_template('result.html', recommendation=recommendation, rating=avg_rating, user_image=pic2)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True,port=5001)
