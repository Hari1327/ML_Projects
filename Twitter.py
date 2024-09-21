import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import re
import base64

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit UI
st.header("Twitter Comment Sentiment Analysis")

# def get_base64_image(image_path):
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode('utf-8')

# base64_image = get_base64_image("bg.png")

# # Create CSS with the base64 image
# background_css = f"""
# <style>
# body {{
#     background-image: url('data:image/png;base64,{base64_image}');
#     background-size: cover;
#     background-position: center;
#     background-repeat: no-repeat;
#     background-attachment: fixed;
# }}
# </style>
# """

# Apply the CSS
# st.markdown(background_css, unsafe_allow_html=True)

user_input = st.text_input("Enter your comment here")

with open('Twitter Sentiment Analysis/logistic_regression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

# Preprocess the new data
def preprocess(text):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9 ]+', ' ', text)
    return text

new_data = [preprocess(user_input)]

# Transform the new data using CountVectorizer
# Make sure to use the same vectorizer fitted on the training data
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=nltk.corpus.stopwords.words('english'),
    ngram_range=(1, 4)
)

reviews_train =  pd.read_csv("twitter_training.csv")
reviews_train.columns = ['id', 'information', 'type', 'text']

reviews_train["lower"]=reviews_train.text.str.lower() #lowercase
reviews_train["lower"]=[str(data) for data in reviews_train.lower] #converting all to string
reviews_train["lower"]=reviews_train.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex

#train data spilt

reviews_train, reviews_test = train_test_split(reviews_train,test_size=0.2, random_state = 42)
# Assuming `reviews_train` is available from the training phase
X_train_bow = bow_counts.fit_transform(reviews_train.lower)

# Transform the new data
X_new_data_bow = bow_counts.transform(new_data)

# Make predictions using logistic regression
logistic_predictions = logistic_model.predict(X_new_data_bow)
print("Logistic Regression Predictions:", logistic_predictions)


# Display the prediction
st.markdown(f"Prediction: {logistic_predictions}")
