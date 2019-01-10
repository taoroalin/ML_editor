import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
import nltk
from nltk.corpus import words
import re
import random

allwords = words.words()

def insert_generate(sentence):
    bad = sentence.copy()
    bad.insert(random.randint(0, len(sentence)-1), random.choice(allwords))
    return [sentence, bad]

def flip_generate(sentence, flips=1):
    bad = sentence.copy()
    for flip in flips:
        first = random.randint(0, len(sentence)-1)
        second = random.randint(0, len(sentence)-1)
        temp = bad[first]
        bad[first] = bad[second]
        bad[second] = temp
    return [sentence, bad]

def combine(pair):
    return [" ".join(pair[0]), " ".join(pair[1])]

def read(fname):
    tokenized = []
    with open(fname) as file:
        text = file.read()
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    for sentence in sentences:
        if len(sentence) > 4 and len(sentence) < 30:
            tokenized.append(nltk.word_tokenize(sentence))
    return tokenized

def generate_save(fname, nname):
    sentences = read(fname)
    pairs = [combine(insert_generate(sentence)) for sentence in sentences]
    df = pd.dataFrame(pairs, columns=None, headers=None)
    print(df.head())
    df.to_csv(nname)

generate_save("./books/amanobsessed.txt", "./datasets/1.csv")

