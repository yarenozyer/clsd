import re

def parse_turkish_tweets(text, target, stance, native_tweets, native_targets, native_stances):
    cleaned_text = re.sub(r'<ENAMEX TYPE="[^"]+">([^<]+)</ENAMEX>', r'\1', text)
    
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)
    
    cleaned_text = re.sub(r'@\S+', '', cleaned_text)
    cleaned_text = re.sub(r'#\S+', '', cleaned_text)
    
    sentences = re.split(r'\s*\n\s*', cleaned_text)
    sentences = [sentence for sentence in sentences if sentence]
    targets= [target for sentence in sentences if sentence]
    stances = [stance for sentence in sentences if sentence]
    native_tweets.extend(sentences)
    native_targets.extend(targets)
    native_stances.extend(stances)
