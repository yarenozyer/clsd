import numpy as np
from nltk.tokenize import word_tokenize
from snowballstemmer import TurkishStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

#stemmer = TurkishStemmer()

def read_turkish_tweets(file):
    df = pd.read_csv(file, encoding='windows-1254')
    tweets = df["Tweet"].tolist()
    targets = df["Target"].tolist()
    return tweets, targets
        
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
    stop_words = detect_stopwords()
    normalized_tokens = [token.lower() for token in tokens]
    filtered_tokens = [token for token in normalized_tokens if (token not in stop_words and not token.startswith("http"))]
    #stemmed_tokens = [stemmer.stemWord(token) for token in filtered_tokens]
    
    return filtered_tokens

def extract_features_tfidf(tweets):
    # Tokenization and preprocessing
    tokenized_tweets = [tokenize_tweet(tweet) for tweet in tweets]
    
    # Feature extraction: n-grams
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
    tfidf_features = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_tweets])
    
    # Feature extraction: character n-grams
    char_tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 5), analyzer='char')
    char_tfidf_features = char_tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_tweets])
    
    # Feature extraction: sentiment lexicon features, target presence/absence, POS tags, encodings
    # These features remain the same as before
    
    # Combine all features
    all_features = np.concatenate((tfidf_features.toarray(), char_tfidf_features.toarray()), axis=1)
    np.savetxt('feature_matrix.csv', all_features, delimiter=',')
    return all_features

train_tweets, train_targets = read_turkish_tweets('translated_train_without_none.csv')

print(extract_features_tfidf(train_tweets))

#print(tokenize_tweet(train_tweets[0]))