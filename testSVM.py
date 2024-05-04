import numpy as np
from nltk.tokenize import word_tokenize
from snowballstemmer import TurkishStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from zemberek import TurkishSpellChecker, TurkishMorphology, TurkishSentenceNormalizer

stemmer = TurkishStemmer()

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)
spell_checker = TurkishSpellChecker(morphology)

favor = []
against = []
predicted = []

def detect_stopwords():
    stopwords_df = pd.read_csv('turkish', header=None)
    stop_words = stopwords_df[0].tolist()
    stop_words.extend(string.punctuation)
    stop_words.extend(["vs.", "vb.", "a", "i", "e", "rt", "#semst", "semst"])
    stop_words = set(stop_words)
    return stop_words
    
    
def tokenize_tweet(tweet):
    tweet = normalizer.normalize(tweet)
    tokens = word_tokenize(tweet)
    stop_words = detect_stopwords()
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
    
def t_tweets(file, encoding):
    df = pd.read_csv(file, encoding = encoding)
    
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
    # precision = precision_score(substances_test, stance_pred, pos_label=)
    
    # f1_positive = (2 * precision * recall ) / (precision + recall)
    
    f_macro = f1_score(substances_test, stance_pred, average='macro')
    f1_positive = f1_score(substances_test, stance_pred, average=None)[0]  # Positive class
    f1_negative = f1_score(substances_test, stance_pred, average=None)[1]  # Negative class
    f1_none = f1_score(substances_test, stance_pred, average=None)[2]  # Negative class
    predicted.extend(stance_pred)
    favor.append(f1_positive)
    against.append(f1_negative)
    print(target + " Accuracy:", accuracy*100)
    print(target + " F Macro: ", f_macro*100)
    print(target + " F1-Score (Negative Class):", f1_positive * 100)
    print(target + " F1-Score (Positive Class):", f1_negative * 100)
    print(target + " F1-Score (None Class):", f1_none * 100)
def svm_all_targets(tweets_train, tweets_test, stances_train, stances_test, targets):
        
    print("Training")
    tokenized_train = [tokenize_tweet(tweet) for tweet in tweets_train]
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer='word')
    word_train_features = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_train])

    #char_tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 5), analyzer='char')
    #char_train_features = char_tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_train])
    
    train_features = word_train_features.toarray()
    
    svm_classifier = SVC(kernel='sigmoid', C=10)
    svm_classifier.fit(train_features, stances_train)
    
    print("Evaluating Results")
    for target in targets:
        subtweets = tweets_test[target]
        substances= stances_test[target]
        tokenized_subtest = [tokenize_tweet(tweet) for tweet in subtweets]
        word_test_features = tfidf_vectorizer.transform([' '.join(tokens) for tokens in tokenized_subtest])
        #char_test_features = char_tfidf_vectorizer.transform([' '.join(tokens) for tokens in tokenized_subtest])
        
        test_features = word_test_features.toarray()
        
        stance_pred = svm_classifier.predict(test_features)
        accuracy = accuracy_score(substances, stance_pred)
        f_macro = f1_score(substances, stance_pred, average='macro')
        f1_positive = f1_score(substances, stance_pred, average=None)[0]  # Positive class
        f1_negative = f1_score(substances, stance_pred, average=None)[1]  # Negative class
        f1_none = f1_score(substances, stance_pred, average=None)[2]  # Negative class
    
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


def translate(texts, model, tokenizer, language):
    """Translate texts into a target language"""
    # Format the text as expected by the model
    
    formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
    original_texts = [formatter_fn(txt) for txt in texts]

    # Tokenize (text to tokens)
    inputs = tokenizer(original_texts, return_tensors="pt", padding=True, truncation=True)

    # Translate
    translated = model.generate(**inputs)

    # Decode (tokens to text)
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return translated_texts

def download(model_name):
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  return tokenizer, model
   

# download model for English -> Romance
#tmp_lang_tokenizer, tmp_lang_model = download("Helsinki-NLP/opus-mt-en-trk")


tweets_train, stances_train, all_tweets_train, all_stances_train = t_tweets('IBM_train.csv', "ANSI")

tweets_test, stances_test, all_tweets_test, all_stances_test = t_tweets('IBM_test.csv', "ANSI")

svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Atheism")
svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Climate Change is a Real Concern")
svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Feminist Movement")
svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Hillary Clinton")
svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Legalization of Abortion")

precision_pos = precision_score(all_stances_test, predicted, labels=["FAVOR"], average="macro")
recall_pos = recall_score(all_stances_test, predicted, labels=["FAVOR"], average="macro")
f_calculated_pos = (2*precision_pos*recall_pos) /(precision_pos + recall_pos)

precision_neg = precision_score(all_stances_test, predicted, labels=["AGAINST"], average="macro")
recall_neg = recall_score(all_stances_test, predicted, labels=["AGAINST"], average="macro")
f_calculated_neg = (2*precision_neg*recall_neg) /(precision_neg + recall_neg)
    
print("F_FAV = ", f_calculated_pos *100)
print("F_NEG = ", f_calculated_neg*100)
print("F_AVG = ", (f_calculated_pos + f_calculated_neg)*50)

# targets = ["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"]
# svm_all_targets(all_tweets_train, tweets_test , all_stances_train, stances_test, targets)
