import numpy as np
from nltk.tokenize import word_tokenize
from snowballstemmer import TurkishStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import nltk

stemmer = TurkishStemmer()

def detect_stopwords():
    #print("Detecting stopwords")
    stopwords_df = pd.read_csv('turkish', header=None)
    stop_words = stopwords_df[0].tolist()
    #stop_words = stopwords.words('turkish')
    stop_words.extend(string.punctuation)
    stop_words.extend(["vs.", "vb.", "a", "i", "e", "rt", "#semst", "semst"])
    stop_words = set(stop_words)
    return stop_words
    
    
def tokenize_tweet(tweet):
    tokens = word_tokenize(tweet)
    stop_words = []
    normalized_tokens = [token.lower() for token in tokens]
    filtered_tokens = [token for token in normalized_tokens if (token not in stop_words and not token.startswith("http"))]
    #filtered_tokens = [stemmer.stemWord(token) for token in filtered_tokens]
    
    return filtered_tokens

def extract_features_tfidf_ngram(train_tweets, test_tweets):
    #print("Extracting Features")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
    word_train_features = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in train_tweets])
    word_test_features = tfidf_vectorizer.transform([' '.join(tokens) for tokens in test_tweets])

    char_tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 5), analyzer='char')
    char_train_features = char_tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in train_tweets])
    char_test_features = char_tfidf_vectorizer.transform([' '.join(tokens) for tokens in test_tweets])
    
    return np.concatenate((word_train_features.toarray(), char_train_features.toarray()), axis=1), np.concatenate((word_test_features.toarray(), char_test_features.toarray()), axis=1)
    
def extract_features_tfidf_unigram(train_tweets, test_tweets):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer='word')
    word_train_features = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in train_tweets])
    word_test_features = tfidf_vectorizer.transform([' '.join(tokens) for tokens in test_tweets])
    
    return word_train_features.toarray(), word_test_features.toarray()
    
def t_tweets(file):
    #print("Reading File")
    df = pd.read_csv(file, encoding='windows-1254')
    
    tweets_by_target = {}
    stances_by_target = {}
    all_tweets = []
    all_stances = []
    
    unique_targets = df["Target"].unique()
    for target in unique_targets:
        target_df = df[df["Target"] == target]
        
        tweets_by_target[target] = target_df["Tweet"].tolist()
        stances_by_target[target] = target_df["Stance"].tolist()
        
        all_tweets.extend(target_df["Tweet"].tolist())
        all_stances.extend(target_df["Stance"].tolist())
        
    return tweets_by_target, stances_by_target, all_tweets, all_stances
    
def svm_for_target(tweets_train, stances_train, tweets_test, stances_test, target):
    subtweets_train = tweets_train[target]
    substances_train = stances_train[target]
    subtweets_test = tweets_test[target]
    substances_test = stances_test[target]
    
    tokenized_train = [tokenize_tweet(tweet) for tweet in subtweets_train]
    tokenized_test = [tokenize_tweet(tweet) for tweet in subtweets_test]
    
    train_features, test_features = extract_features_tfidf_ngram(tokenized_train, tokenized_test)

    # Train SVM with the best parameters
    svm_classifier = SVC(kernel='sigmoid', C=10)
    svm_classifier.fit(train_features, substances_train)
    
    #print("Evaluating Results")
    stance_pred = svm_classifier.predict(test_features)
    accuracy = accuracy_score(substances_test, stance_pred)
    f_macro = f1_score(substances_test, stance_pred, average='macro')
    f1_positive = f1_score(substances_test, stance_pred, average=None)[0]  # Positive class
    f1_negative = f1_score(substances_test, stance_pred, average=None)[1]  # Negative class
    
    print(target + " Accuracy:", accuracy*100)
    print(target + " F Macro: ", f_macro*100)
    print(target + " F1-Score (Positive Class):", f1_positive * 100)
    print(target + " F1-Score (Negative Class):", f1_negative * 100)


def svm_all_targets(tweets_train, tweets_test, stances_train, stances_test, targets):
        
    print("Training")
    tokenized_train = [tokenize_tweet(tweet) for tweet in tweets_train]
    #tokenized_test = [tokenize_tweet(tweet) for tweet in tweets_test]
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
    word_train_features = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_train])

    char_tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 5), analyzer='char')
    char_train_features = char_tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_train])
    
    train_features = np.concatenate((word_train_features.toarray(), char_train_features.toarray()), axis=1)
    
    svm_classifier = SVC(kernel='sigmoid', C=10)
    svm_classifier.fit(train_features, stances_train)
    
    print("Evaluating Results")
    for target in targets:
        subtweets = tweets_test[target]
        substances= stances_test[target]
        tokenized_subtest = [tokenize_tweet(tweet) for tweet in subtweets]
        word_test_features = tfidf_vectorizer.transform([' '.join(tokens) for tokens in tokenized_subtest])
        char_test_features = char_tfidf_vectorizer.transform([' '.join(tokens) for tokens in tokenized_subtest])
        
        test_features = np.concatenate((word_test_features.toarray(), char_test_features.toarray()), axis=1)
        
        stance_pred = svm_classifier.predict(test_features)
        accuracy = accuracy_score(substances, stance_pred)
        f_macro = f1_score(substances, stance_pred, average='macro')
        f1_positive = f1_score(substances, stance_pred, average=None)[0]  # Positive class
        f1_negative = f1_score(substances, stance_pred, average=None)[1]  # Negative class
    
        print(f"Combined {target} Accuracy: {accuracy * 100}")
        print(f"Combined {target} F Macro: {f_macro*100}")
        print(f"Combined {target} F1-Score (Positive Class): {f1_positive * 100}")
        print(f"Combined {target} F1-Score (Negative Class): {f1_negative * 100}")
  
def tune_svm(features, stances):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    
    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid_search.fit(features, stances)

    best_params = grid_search.best_params_
    
    svm_classifier = SVC(**best_params)
    return svm_classifier  

tweets_train, stances_train, all_tweets_train, all_stances_train = t_tweets('translated_train_without_none.csv')

tweets_test, stances_test, all_tweets_test, all_stances_test = t_tweets('translated_test_without_none.csv')

# svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Ateizm")
# svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "İklim Değişikliği Gerçek Bir Endişe Kaynağı")
# svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Feminist Hareket")
# svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Hillary Clinton")
# svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Kürtajın Yasallaştırılması")

targets = ["Ateizm", "İklim Değişikliği Gerçek Bir Endişe Kaynağı", "Feminist Hareket", "Hillary Clinton", "Kürtajın Yasallaştırılması"]

svm_all_targets(all_tweets_train, tweets_test , all_stances_train, stances_test, targets)

#print(stemmer.stemWord("istiyorum"))
#print(stemmer.stemWord("istemiyorum"))