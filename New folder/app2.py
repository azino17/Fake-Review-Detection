from flask import Flask, render_template, request, redirect, url_for
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from google_play_scraper import app, reviews_all

app = Flask(__name__)

# Function to scrape reviews from the Google Play Store for a given app
def scrape_reviews(app_id, num_pages=5):
    reviews = []
    for page in range(1, num_pages + 1):
        url = f'https://play.google.com/store/getreviews?id={app_id}&hl=en&pagTok={page}&reviewSortOrder=0&xhr=1'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            soup = BeautifulSoup(data[0][2], 'html.parser')
            review_divs = soup.find_all('div', class_='d15Mdf')
            for review_div in review_divs:
                review_text = review_div.find('span', class_='review-body').text.strip()
                reviews.append(review_text)
    return reviews

# Function to preprocess text data
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Function to train and evaluate the model
def train_and_evaluate_model(app_id):
    # Fetching all reviews for the app
    reviews = reviews_all(app_id)

    # Preprocessing reviews
    preprocessed_reviews = [preprocess_text(review['content']) for review in reviews]

    # Sample labels (0 for genuine, 1 for potentially computer-generated)
    labels = [0 if review['score'] >= 3 else 1 for review in reviews]

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_reviews, labels, test_size=0.2, random_state=42)

    # Vectorizing text data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Training a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Making predictions
    y_pred = classifier.predict(X_test_vectorized)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Counting the number of human-generated and computer-generated reviews
    human_reviews_count = sum(1 for label in y_test if label == 0)
    computer_reviews_count = sum(1 for label in y_test if label == 1)

    return accuracy, classification_rep, human_reviews_count, computer_reviews_count

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        app_id = request.form['app_id']
        return redirect(url_for('results', app_id=app_id))
    return render_template('index.html')

@app.route('/results/<app_id>')
def results(app_id):
    accuracy, classification_rep, human_reviews_count, computer_reviews_count = train_and_evaluate_model(app_id)
    return render_template('results.html', accuracy=accuracy, classification_rep=classification_rep, 
                           human_reviews_count=human_reviews_count, computer_reviews_count=computer_reviews_count)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
