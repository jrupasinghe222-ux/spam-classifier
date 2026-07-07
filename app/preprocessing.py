import string
import re

def preprocess(text:str):
    # turn lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans("","", string.punctuation))

    # remove numbers
    # text = re.sub(r"\d+","",text)

    #remove spaces 
    text = re.sub(r"\s+", " ", text).strip()


    return text