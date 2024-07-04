import numpy as np
from nltk.tokenize import word_tokenize
from snowballstemmer import TurkishStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from zemberek import TurkishMorphology, TurkishSentenceNormalizer
import re


stemmer = TurkishStemmer()

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)

predicted = []

def detect_stopwords():
    stopwords_df = pd.read_csv('turkish', header=None)
    stop_words = stopwords_df[0].tolist()
    stop_words.extend(string.punctuation)
    stop_words.extend(["vs.", "vb.", "a", "i", "e", "rt", "#semst", "semst"])
    stop_words = set(stop_words)
    return stop_words
    
    
def tokenize_tweet_w_pre(tweet):
    tweet = normalizer.normalize(tweet)
    tokens = word_tokenize(tweet)
    stop_words = detect_stopwords()
    normalized_tokens = [token.lower() for token in tokens]
    filtered_tokens = [token for token in normalized_tokens if (token not in stop_words and not token.startswith("http"))]
    filtered_tokens = [stemmer.stemWord(token) for token in filtered_tokens]
    return filtered_tokens

def tokenize_tweet(tweet):
    tokens = word_tokenize(tweet)
    stop_words = detect_stopwords()
    normalized_tokens = [token.lower() for token in tokens]
    filtered_tokens = [token for token in normalized_tokens if (token not in stop_words and not token.startswith("http"))]
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
    
def parse_turkish_tweets(files, targets, stances):
    print("parsing")
    tweets_by_target = {}
    stances_by_target = {}
    all_tweets = []
    all_stances = []
    all_targets = []
    for file_ind in range(len(files)):
        target = targets[file_ind]
        stance = stances[file_ind]
        with open(files[file_ind], 'r', encoding='utf-8') as file:
            text = file.read()
            
        cleaned_text = re.sub(r'<ENAMEX TYPE="[^"]+">([^<]+)</ENAMEX>', r'\1', text)
        cleaned_text = re.sub(r'http\S+', '', cleaned_text)
        cleaned_text = re.sub(r'@\S+', '', cleaned_text)
        cleaned_text = re.sub(r'#\S+', '', cleaned_text)
        sentences = re.split(r'\s*\n\s*', cleaned_text)
        
        sentences = [sentence for sentence in sentences if sentence]
        
        if target not in tweets_by_target:
            tweets_by_target[target] = []
            stances_by_target[target] = []
        
        tweets_by_target[target].extend(sentences)
        stances_by_target[target].extend([stance] * len(sentences))

        all_tweets.extend(sentences)
        all_stances.extend([stance] * len(sentences))
        all_targets.extend([target] * len(sentences))
        
        
    return all_tweets, all_targets, all_stances, tweets_by_target, stances_by_target
    

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

    svm_classifier = SVC(kernel='sigmoid', C=10)
    svm_classifier.fit(train_features, substances_train)
   # svm_classifier = tune_svm(train_features, substances_train)
    
    #print("Evaluating Results")
    stance_pred = svm_classifier.predict(test_features)
    accuracy = accuracy_score(substances_test, stance_pred)
    
    f_macro = f1_score(substances_test, stance_pred, average='macro')
    
    predicted.extend(stance_pred)
    print(target + " Accuracy:", accuracy*100)
    print(target + " F Macro: ", f_macro*100)
    
def svm_all_targets(tweets_train, tweets_test, stances_train, stances_test, targets):
        
    print("Training")
    tokenized_train = [tokenize_tweet(tweet) for tweet in tweets_train]
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
        predicted.extend(stance_pred)
        print(f"Combined {target} Accuracy: {accuracy * 100}")
        print(f"Combined {target} F Macro: {f_macro*100}")
  
def tune_svm(features, stances):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'degree': [2, 3, 4],
        'coef0': [0.0, 0.1, 0.5]
    }

    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid_search.fit(features, stances)

    best_params = grid_search.best_params_
    
    svm_classifier = SVC(**best_params)
    return svm_classifier  
   
def run_svm_for_english_targets(tweets_train, stances_train, tweets_test, stances_test):
    svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Atheism")
    svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Climate Change is a Real Concern")
    svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Feminist Movement")
    svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Hillary Clinton")
    svm_for_target(tweets_train, stances_train, tweets_test, stances_test, "Legalization of Abortion")
    
def run_svm_for_all_targets(all_tweets_train, all_stances_train, tweets_test, stances_test, targets):
    svm_all_targets(all_tweets_train, tweets_test , all_stances_train, stances_test, targets)
    
def svm_cross_validation(tweets_train, stances_train, native_tweets, native_stances, native_targets):
    
    tokenized_train = [tokenize_tweet_w_pre(tweet) for tweet in tweets_train]
    tokenized_native = [tokenize_tweet(tweet) for tweet in native_tweets]
    train_features, native_features = extract_features_tfidf_ngram(tokenized_train, tokenized_native)
    native_stances = np.array(native_stances)
    native_targets = np.array(native_targets)
    print("Training")
    svm_classifier = SVC(kernel='sigmoid', C=10)
    svm_classifier.fit(train_features, stances_train)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    n=0
    macros=[]
    macros_fener=[]
    macros_galata=[]
    for train_index, test_index in skf.split(native_features, native_stances):
        
        print(f"FOLD {n}")
        print("train index: ", train_index)
        print("test index: ", test_index)
        predicted=[]
        X_train, X_test = native_features[train_index], native_features[test_index]
        y_train, y_test = native_stances[train_index], native_stances[test_index]
        t_test =  native_targets[test_index]
        
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)
        f_macro = f1_score(y_test, y_pred, average='macro')
        predicted.extend(y_pred)
        print(f"Combined {n} F Macro: {f_macro*100}")
        macros.append(f_macro)
        
        unique_targets = np.unique(t_test)
        for target in unique_targets:
            target_mask = t_test == target
            target_y_test = y_test[target_mask]
            target_y_pred = y_pred[target_mask]
            target_f_macro = f1_score(target_y_test, target_y_pred, average='macro')
            print(f"Target '{target}' F Macro: {target_f_macro * 100:.2f}")
            if(target == "Fenerbahçe"):
                macros_fener.append(target_f_macro)
            else:
                macros_galata.append(target_f_macro)
        n+=1
    
    print("AVG", np.mean(macros))
    print("FENER", np.mean(macros_galata))
    print("GALATA", np.mean(macros_fener))

        
def svm_cross_validation_deneme(tweets_train, stances_train, native_tweets, native_stances, native_targets):
    
    tokenized_train = [tokenize_tweet_w_pre(tweet) for tweet in tweets_train]
    tokenized_native = [tokenize_tweet(tweet) for tweet in native_tweets]
    train_features, native_features = extract_features_tfidf_ngram(tokenized_train, tokenized_native)
    native_stances = np.array(native_stances)
    native_targets = np.array(native_targets)
    print("Training")
    svm_classifier = SVC(kernel='sigmoid', C=10)
    svm_classifier.fit(train_features, stances_train)
    
    X_train, X_test, y_train, y_test, _, t_test = train_test_split(
    native_features, native_stances, native_targets, test_size=0.9, stratify=native_stances, random_state=42
)
    macros=[]
    macros_fener=[]
    macros_galata=[]
        
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    f_macro = f1_score(y_test, y_pred, average='macro')
    predicted.extend(y_pred)
    macros.append(f_macro)
            
    unique_targets = np.unique(t_test)
    for target in unique_targets:
        target_mask = t_test == target
        target_y_test = y_test[target_mask]
        target_y_pred = y_pred[target_mask]
        target_f_macro = f1_score(target_y_test, target_y_pred, average='macro')
        print(f"Target '{target}' F Macro: {target_f_macro * 100:.2f}")
        if(target == "Fenerbahçe"):
            macros_fener.append(target_f_macro)
        else:
            macros_galata.append(target_f_macro)
    
    print("AVG", np.mean(macros))
    print("FENER", np.mean(macros_galata))
    print("GALATA", np.mean(macros_fener))
            
def show_results(all_stances_test, predicted):
    precision_pos = precision_score(all_stances_test, predicted, labels=["FAVOR"], average="macro")
    recall_pos = recall_score(all_stances_test, predicted, labels=["FAVOR"], average="macro")
    f_calculated_pos = (2*precision_pos*recall_pos) /(precision_pos + recall_pos)

    precision_neg = precision_score(all_stances_test, predicted, labels=["AGAINST"], average="macro")
    recall_neg = recall_score(all_stances_test, predicted, labels=["AGAINST"], average="macro")
    f_calculated_neg = (2*precision_neg*recall_neg) /(precision_neg + recall_neg)
        
    print("F_FAV = ", f_calculated_pos *100)
    print("F_NEG = ", f_calculated_neg*100)
    print("F_AVG = ", (f_calculated_pos + f_calculated_neg)*50)
    
files = ['fenerbahce_against_v3_answer_1.txt', 'fenerbahce_favor_v3_answer_1.txt', 'galatasaray_against_v3_answer_1.txt', 'galatasaray_favor_v3_answer_1.txt']
file_targets = ['Fenerbahçe', 'Fenerbahçe', 'Galatasaray', 'Galatasaray']
file_stances = ['AGAINST', 'FAVOR', 'AGAINST', 'FAVOR']
    
all_native_tweets, all_native_targets, all_native_stances, native_tweets_by_target, native_stances_by_target = parse_turkish_tweets(files, file_targets, file_stances)
tweets_train, stances_train, all_tweets_train, all_stances_train = t_tweets('gpt_train.csv', "ANSI")
#tweets_test, stances_test, all_tweets_test, all_stances_test = t_tweets('google_translate_test.csv', "ANSI")

#targets = ["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"]
targets = ["Fenerbahçe", "Galatasaray"]
#run_svm_for_all_targets(all_tweets_train, all_stances_train, native_tweets_by_target, native_stances_by_target, targets)
#show_results(all_native_stances, predicted)

svm_cross_validation(all_tweets_train, all_stances_train, all_native_tweets, all_native_stances, all_native_targets)
#svm_cross_validation_deneme(["merhaba", "sgfh"], ["FAVOR", "AGAINST"], all_native_tweets, all_native_stances, all_native_targets)