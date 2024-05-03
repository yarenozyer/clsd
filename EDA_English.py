import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random
import pandas as pd


    

def create_augmented_dataset(file_path):
    # Read existing data from the CSV file
    existing_data = pd.read_csv(file_path, encoding = 'ANSI')
    all_tweets = existing_data["Tweet"].to_list()
    all_targets = existing_data["Target"].to_list()
    all_stances = existing_data["Stance"].to_list()
    alpha = 0.05
    new_data =[]
    new_tweets = []
    new_targets = []
    new_stances = []
    tweet_count = len(all_tweets)
    for i in range(tweet_count):
        new_tweets.append(eda_SR(all_tweets[i], int(len(all_tweets[i])*alpha)))
        new_targets.append(all_targets[i])
        new_stances.append(all_stances[i])
        
    new_data.append(new_tweets)
    new_data.append(new_targets)
    new_data.append(new_stances)
    # Convert new data to a DataFrame
    new_data_df = pd.DataFrame(new_data, columns=existing_data.columns)

    # Concatenate existing data with new data
    combined_data = pd.concat([existing_data, new_data_df], ignore_index=True)

    # Write the combined data back to the CSV file
    combined_data.to_csv(file_path, index=False)


def eda_SR(originalSentence, n):

  stops = set(stopwords.words('english'))
  splitSentence = list(originalSentence.split(" "))
  splitSentenceCopy = splitSentence.copy()
  
  ls_nonStopWordIndexes = []
  for i in range(len(splitSentence)):
    if splitSentence[i].lower() not in stops:
      ls_nonStopWordIndexes.append(i)
  if (n > len(ls_nonStopWordIndexes)):
    raise Exception("The number of replacements exceeds the number of non stop word words")
  for i in range(n):
    indexChosen = random.choice(ls_nonStopWordIndexes)
    ls_nonStopWordIndexes.remove(indexChosen)
    synonyms = []
    originalWord = splitSentenceCopy[indexChosen]
    for synset in wordnet.synsets(originalWord):
      for lemma in synset.lemmas():
        if lemma.name() != originalWord:
          synonyms.append(lemma.name())
    if (synonyms == []):
      continue
    splitSentence[indexChosen] = random.choice(synonyms).replace('_', ' ')
  return " ".join(splitSentence)
     

print(eda_SR("I love to play football", 2))

def eda_RI(originalSentence, n):
  stops = set(stopwords.words('english'))
  splitSentence = list(originalSentence.split(" "))
  splitSentenceCopy = splitSentence.copy() 
  # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
  ls_nonStopWordIndexes = []
  for i in range(len(splitSentence)):
    if splitSentence[i].lower() not in stops:
      ls_nonStopWordIndexes.append(i)
  if (n > len(ls_nonStopWordIndexes)):
    raise Exception("The number of replacements exceeds the number of non stop word words")
  WordCount = len(splitSentence)
  for i in range(n):
    indexChosen = random.choice(ls_nonStopWordIndexes)
    ls_nonStopWordIndexes.remove(indexChosen)
    synonyms = []
    originalWord = splitSentenceCopy[indexChosen]
    for synset in wordnet.synsets(originalWord):
      for lemma in synset.lemmas():
        if lemma.name() != originalWord:
          synonyms.append(lemma.name())
    if (synonyms == []):
      continue
    splitSentence.insert(random.randint(0,WordCount-1), random.choice(synonyms).replace('_', ' '))
  return " ".join(splitSentence)
     



def eda_RS(originalSentence, n):
    
  splitSentence = list(originalSentence.split(" "))
  WordCount = len(splitSentence)
  for i in range(n):
    firstIndex = random.randint(0,WordCount-1)
    secondIndex = random.randint(0,WordCount-1)
    while (secondIndex == firstIndex and WordCount != 1):
      secondIndex = random.randint(0,WordCount-1)
    splitSentence[firstIndex], splitSentence[secondIndex] = splitSentence[secondIndex], splitSentence[firstIndex]
  return " ".join(splitSentence)
     



def eda_RD(originalSentence, p):
  og = originalSentence
  if (p == 1):
      raise Exception("Always an Empty String Will Be Returned") 
  if (p > 1 or p < 0):
    raise Exception("Improper Probability Value")
  splitSentence = list(originalSentence.split(" "))
  lsIndexesRemoved = []
  WordCount = len(splitSentence)
  for i in range(WordCount):
    randomDraw = random.random()
    if randomDraw <= p:
      lsIndexesRemoved.append(i)
  lsRetainingWords = []
  for i in range(len(splitSentence)):
    if i not in lsIndexesRemoved:
      lsRetainingWords.append(splitSentence[i])
  if (lsRetainingWords == []):
    return og
  return " ".join(lsRetainingWords)


# Path to your CSV file
csv_file_path = "aug_train.csv"

create_augmented_dataset(csv_file_path)
     
     
