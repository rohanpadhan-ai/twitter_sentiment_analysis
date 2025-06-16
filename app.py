import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
import joblib

# NLTK setup (download stopwords if needed)
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
tfidf = joblib.load("tfidf_vectorizer2.joblib")
mdl = joblib.load("sentiment_model2.joblib")

# Preprocessing function
def preprocessing_tweets(tweets):
    processed_tweets = []
    for tweet in tweets:
        tweet = re.sub(r'[^a-zA-Z]', ' ', tweet)         # Remove non-alphabetic characters
        tweet = tweet.lower()                            # Convert to lowercase
        words = [word for word in tweet.split() if word not in stop_words]  # Remove stopwords
        stemmed_words = [stemmer.stem(word) for word in words]              # Stemming
        final_tweet = ' '.join(stemmed_words)
        processed_tweets.append(final_tweet)
    return processed_tweets

# Sentiment prediction
def predict_sentiment(text, model, vectorizer):
    processed = preprocessing_tweets([text])
    x = vectorizer.transform(processed).toarray()
    pred = model.predict(x)[0]  # This will be 'Positive', 'Negative', or 'Neutral'
    return pred

# ---------------- Streamlit UI ----------------
st.title("Tweet Sentiment Analyzer")

tweet_input = st.text_area("Enter a tweet/text here")

if st.button("Predict Sentiment"):
    if tweet_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        sentiment = predict_sentiment(tweet_input, mdl, tfidf)
        st.success(f"Predicted Sentiment: {sentiment}")
