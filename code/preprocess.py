import numpy as np
import spacy
import pandas as pd
import subprocess
from langdetect import detect
import re

## install english spacy model (if not already installed) and load
try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load("en_core_web_sm")
    
## define function to preprocess
def preprocess_text(text):
    """
    Preprocesses the given text by performing the following steps:
    1. Detects the language of the text using a language detection library.
    2. Filters out non-English text; returns (pd.NA, pd.NA) if the detected language is not English ('en').
    3. Processes the text using spaCy in batch mode:
       - Converts the text to lowercase.
       - Tokenizes the text.
       - Filters out stop words, URLs, and non-alphabetic tokens.
       - Creates a cleaned version of the text without stop words and URLs.
       - Creates a lemmatized version of the cleaned text.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    tuple: A tuple containing two strings:
        - cleaned (str): The cleaned version of the input text.
        - lemmatized (str): The lemmatized version of the cleaned text.
        Returns (pd.NA, pd.NA) if the language is not English or if an error occurs during processing.
    """
    try:
        ## detect language
        lang = detect(text)
        
        ## keep only english text, return NA otherwise
        if lang != 'en':
            return pd.NA, pd.NA
        
        ## to lower case
        text = text.lower()
        
        ## remove html junk
        text = text.replace('&gt;', ' ')
        text = re.sub(r"u/\w+", "username" , text)
        
        ## disable parser and named entity recognition to safe time, process in batches
        doc = list(nlp.pipe([text], disable=["ner", "parser"]))[0]
        
        ## use list of stopwords for faster lookup, add "gt" which is some html junk
        stop_words = nlp.Defaults.stop_words
        
        cleaned_tokens = []
        for token in doc:
            if token.is_alpha and not token.like_url and token not in stop_words:
                cleaned_tokens.append(token)
        
        cleaned = ' '.join([token.text for token in cleaned_tokens])
        lemmatized = ' '.join([token.lemma_ for token in cleaned_tokens])
        
        return cleaned, lemmatized
    except:
        return "", ""
    

## load data in, drop non relevant cols, can be later merged on id
posts = pd.read_csv("data/raw/posts_sample.csv")[["id", "title", "selftext"]]
comments = pd.read_csv("data/raw/comments_sample.csv")[["id", "body"]]

## preprocess text, to lower case, keep only alphabetical text, lemmatise
comments[['cleaned', 'lemmatized']] = comments['body'].apply(lambda x: pd.Series(preprocess_text(x)))
posts[['title_cleaned', 'title_lemmatized']] = posts['title'].apply(lambda x: pd.Series(preprocess_text(x)))
posts[['selftext_cleaned', 'selftext_lemmatized']] = posts['selftext'].apply(lambda x: pd.Series(preprocess_text(x)))

## combine title and text
## selftext to empty string if nan
posts["selftext"] = posts["selftext"].apply(lambda x: "" if pd.isna(x) else x)
posts["title_and_text"] = posts["title"] + " " + posts["selftext"]
posts["title_and_text_cleaned"] = posts["title_cleaned"] + " " + posts["selftext_cleaned"]
posts["title_and_text_lemmatized"] = posts["title_lemmatized"] + " " + posts["selftext_lemmatized"]


## transform timestamp to human readable
#comments["created_utc"] = pd.to_datetime(comments["created_utc"], unit='s')
#posts["created_utc"] = pd.to_datetime(posts["created_utc"], unit='s')

## write
comments.to_csv("data/preprocessed/comments.csv")
posts.to_csv("data/preprocessed/posts.csv")



## create one df which contains all text
## change col names to enable merge
comments = comments.rename(columns={
    'body': 'title_and_text',
    'cleaned': 'title_and_text_cleaned',
    'lemmatized': 'title_and_text_lemmatized'
})

all_text = pd.concat([
    posts[["title_and_text", "title_and_text_cleaned", "title_and_text_lemmatized"]],
    comments[["title_and_text", "title_and_text_cleaned", "title_and_text_lemmatized"]] ],
                     axis=0).reset_index(drop=True)




all_text.to_csv("data/preprocessed/all_text.csv")