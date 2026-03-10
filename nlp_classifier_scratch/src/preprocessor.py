import string
import re
import nltk
# import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from typing import List,Union
from nltk.corpus import stopwords
import yaml

def setup_nltk():
    import nltk
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for r in resources:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r)





lemmatizer =WordNetLemmatizer()
stopwords=set(stopwords.words('english'))







def clean_string(text:str):
    text=text.lower()
    text=re.sub(r'https\S+|www\S+|http\S+','',text)
    text=re.sub(r'\S+@\S+','',text)
    text=re.sub(r'<.*?>','',text)
    text=text.strip()
    return text


def tokenise(text:str)->list[str]:
    return word_tokenize(text=text)


def remove_punctuation(text:list[str])->list[str]:
    return[tokens for tokens in text if tokens not in string.punctuation ]


def remove_stopwords(tokens:list[str])->list[str]:
    return [token for token in tokens if token not in stopwords]


def lemmatize_nltk(tokens:list[str])->list[str]:
    return[lemmatizer.lemmatize(token) for token in tokens]


def filter_short_words(tokens:list[str])->list[str]:
    return[token for token in tokens if len(token)>=2]


def preprocess(text:str)->str:
    text=clean_string(text)

    tokens=tokenise(text)
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    tokens=lemmatize_nltk(tokens)
    tokens=filter_short_words(tokens)
    return ''.join(tokens)

def batch_preprocess(texts: List[str], show_progress: bool = True) -> List[str]:
    from tqdm import tqdm
    if show_progress:
        return [preprocess(text) for text in tqdm(texts, desc="data preprocessing...")]
    else:
        return [preprocess(text) for text in texts]

        
    




if __name__ == "__main__":
# Demo usage
    setup_nltk()

    sample_text = """
    Hello! This is a SAMPLE text with URLs like https://example.com 
    and emails like test@email.com. It has punctuation!!! And stopwords.
    We're testing the preprocessing pipeline here.
    """

    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")

    processed = batch_preprocess(sample_text)
    print("Processed text:")
    print(processed)














  