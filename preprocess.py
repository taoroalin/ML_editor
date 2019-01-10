import torch
import torch.nn as nn
import torchtext
import torchtext.data as data

def preprocess(fname):
    dataset = data.TabularDataset(
           path='fname', format='csv',
           fields={'sentence_tokenized': ('text', data.Field(sequential=True)),
                   'noisy_sentence_tokenized': ('text', data.Field(sequential=True))})
    dataset.noisy_sentence_tokenized.build_vocab(dataset, max_size=80000)
    dataset.sentence_tokenized.build_vocab(dataset, max_size=80000)

    return dataset