import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit UI
st.header("Twitter Comment Sentiment Analysis")

page_bg_img = '''
<style>
body {
background-image: url();
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
user_input = st.text_input("Enter your comment here")

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9 ]+', ' ', text)
    return text

# Load training data for CountVectorizer
reviews_train = pd.read_csv("twitter_training.csv")
reviews_train.columns = ['id', 'information', 'type', 'text']
reviews_train["lower"] = reviews_train.text.str.lower()  # Lowercase
reviews_train["lower"] = [str(data) for data in reviews_train.lower]  # Convert all to string
reviews_train["lower"] = reviews_train.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))  # Regex

# Train-test split
reviews_train, reviews_test = train_test_split(reviews_train, test_size=0.2, random_state=42)

# Vectorize the training data
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=nltk.corpus.stopwords.words('english'),
    ngram_range=(1, 4)
)
X_train_bow = bow_counts.fit_transform(reviews_train.lower)

if user_input:
    # Preprocess the user input
    new_data = [preprocess(user_input)]
    
    # Transform the new data using the same vectorizer
    X_new_data_bow = bow_counts.transform(new_data)
    
    # Make predictions using logistic regression
    logistic_predictions = logistic_model.predict(X_new_data_bow)
    
    # Display the prediction
    st.markdown(f"Prediction: {'Positive' if logistic_predictions[0] == 1 else 'Negative'}")
