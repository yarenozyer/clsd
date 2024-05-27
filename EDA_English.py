from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random
import pandas as pd

stop_words = set(stopwords.words('english'))
def create_augmented_dataset(file_path, output_path):
    # Read existing data from the CSV file
    existing_data = pd.read_csv(file_path, encoding='ANSI')
    all_tweets = existing_data["Tweet"].to_list()
    all_targets = existing_data["Target"].to_list()
    all_stances = existing_data["Stance"].to_list()
    alpha = 0.05
    new_tweets = []
    new_targets = []
    new_stances = []
    operations = []
    tweet_count = len(all_tweets)

    operation_choice = ['SR', 'RI', 'RS', 'RD']

    for i in range(tweet_count):
        original_tweet = all_tweets[i]
        split_sentence = original_tweet.split()
        split_sentence_copy = split_sentence.copy()
        n = round(len(original_tweet.split()) * alpha)
        for op_index in range(8):
            operation = operation_choice[op_index % 4]
            if operation == 'SR':
                new_tweet = eda_SR(split_sentence_copy, n)
            elif operation == 'RI':
                new_tweet = eda_RI(split_sentence_copy, n)
            elif operation == 'RS':
                new_tweet = eda_RS(split_sentence_copy, n)
            elif operation == 'RD':
                new_tweet = eda_RD(split_sentence_copy, alpha)
            
            new_tweets.append(new_tweet)
            new_targets.append(all_targets[i])
            new_stances.append(all_stances[i])
            operations.append(operation)

    # Combine new data into a DataFrame
    new_data_df = pd.DataFrame({
        "Tweet": new_tweets,
        "Target": new_targets,
        "Stance": new_stances,
        "Operation": operations
    })
    

    # Write the new data to a new CSV file
    new_data_df.to_csv(output_path, index=False)
            

def eda_SR(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    return sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def eda_RD(words, p):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)
    return sentence


def eda_RS(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
        
    sentence = ' '.join(new_words)
    return sentence

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

def eda_RI(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
        
    sentence = ' '.join(new_words)
    return sentence

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
 
 
# Paths to your CSV files
input_csv_file_path = "train.csv"
output_csv_file_path = "augmented_training_data_4_005.csv"

create_augmented_dataset(input_csv_file_path, output_csv_file_path)


