import numpy as np
from nltk.tokenize import word_tokenize
from snowballstemmer import TurkishStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import nltk

#stemmer = TurkishStemmer()

print("wtfff")
def read_turkish_tweets(file):
    df = pd.read_csv(file, encoding='windows-1254')
    tweets = df["Tweet"].tolist()
    targets = df["Target"].tolist()
    stances = df["Stance"].tolist()
    return tweets, stances, targets
        
def detect_stopwords():
    stopwords_df = pd.read_csv('turkish', header=None)
    stop_words = stopwords_df[0].tolist()
    #stop_words = stopwords.words('turkish')
    stop_words.extend(string.punctuation)
    stop_words.extend(["vs.", "vb.", "a", "i", "e", "rt", "#semst", "semst"])
    stop_words = set(stop_words)
    return stop_words
    
    
def tokenize_tweet(tweet):
    # Tokenization
    tokens = word_tokenize(tweet)
    stop_words = []
    normalized_tokens = [token.lower() for token in tokens]
    filtered_tokens = [token for token in normalized_tokens if (token not in stop_words and not token.startswith("http"))]
    #stemmed_tokens = [stemmer.stemWord(token) for token in filtered_tokens]
    
    return filtered_tokens

def extract_features_tfidf(tweets):
    
    # Feature extraction: n-grams
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
    tfidf_features = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in tweets])
    
    return tfidf_features.toarray()

train_tweets, train_labels, train_targets = read_turkish_tweets('translated_train_without_none.csv')

tokenized_tweets = [tokenize_tweet(tweet) for tweet in train_tweets]

features = extract_features_tfidf(tokenized_tweets)

X_train, X_test, y_train, y_test = train_test_split(features, train_labels, test_size=0.2, random_state=42)

# Step 2: Train the SVM classifier
svm_classifier = SVC(kernel='linear')  # You can choose different kernels based on your data
svm_classifier.fit(X_train, y_train)

# Step 3: Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)