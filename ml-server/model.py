import spacy as sp
import pandas as pd
import numpy as np
import spacy as sp
import torch
from torch import nn
from torch import optim
import random
import torch.nn.functional as F
from sklearn import preprocessing
import sklearn.metrics.pairwise as pairwise
from datetime import datetime, date, time
import re
import codecs
from sklearn.metrics import accuracy_score
from collections import Counter
import string 
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support
import pickle
import os

currDir = os.path.dirname(os.path.realpath(__file__))

class Vocab:
    def __init__(self):
        self.word_count = 0
        self.ind2word = {}
        self.word2ind = {}
        
        self.tag_count = 0
        self.tag2id = {}
        self.id2tag = {}
    
    # Gets a sentence and indexes each word in it
    def index_document(self, document):
        indexes = []
        for token in document:
          result = self.index_token(token)
          if(result != None):
            indexes.append(result)
        return indexes

    # Gets a word, and only if it wasn't previously indexed, it indexes it
    # in two arrays
    def index_token(self, token):
        if token in self.word2ind:
          return self.word2ind[token]
          #self.word2ind[token] = self.word_count
          #self.ind2word[self.word_count] = token
          #self.word_count += 1
    
    # For each tag (Positive, negative, neutral) - returns an id (0, 1, 2)
    def get_tag_id(self, tag):
        if tag not in self.tag2id:
            self.tag2id[tag] = self.tag_count
            self.id2tag[self.tag_count] = tag
            self.tag_count += 1
        return self.tag2id[tag]
		

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size
        
        # The linear layer's input is 2*hidden, and the output is 2*hidden
        self.lin = nn.Linear(self.hidden_size*2, hidden_size*2)
        
        # The weight vector is 2*hidden
        self.weight_vec = nn.Parameter(torch.FloatTensor(1, hidden_size*2))

    def forward(self, outputs):
        seq_len = len(outputs)
        
        # Zeorizing the energies
        attn_energies = torch.zeros(seq_len)

        # For every token in the tweet
        for i in range(seq_len):
            attn_energies[i] = self.score(outputs[i])

        # All the energies    
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, output):
      
        # Feeding the linear layer with one token from the whole tweet
        energy = self.lin(output)
        energy = torch.dot(self.weight_vec.view(-1), energy.view(-1))
        return energy	


class EmoModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, output_size, n_layers):
        super(EmoModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        # Gets the word count and the embedding size
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Input size - embedding size. Output size - hidden size.
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional=True)
        self.out = nn.Linear(hidden_size*2, output_size)
        self.attn = Attn(hidden_size)
    
    def forward(self, input_text):
      
        # The number of words in the current tweet.
        seq_len = len(input_text.data)
        
        embedded_words = self.embedding(input_text).view(seq_len, 1, -1)

        # Zeorizing
        last_hidden = self.init_hidden()
        
        # Going through bi-directional LSTM
        # Returns RNN_outputs (tweet length, 1 batch, 2*hidden size)
        rnn_outputs, hidden = self.lstm(embedded_words, last_hidden)
        
        # Attention model on the outputs of the LSTM
        # Returns the weights.
        attn_weights = self.attn(rnn_outputs)
        
        # Reshaping
        attn_weights = attn_weights.squeeze(1).view(seq_len, 1)
        
        # Removing the batch dimension (tweet length, 2*hidden size)
        rnn_outputs = rnn_outputs.squeeze(1)
        
        attn_weights = attn_weights.expand(seq_len, self.hidden_size*2)
        
        # Multiplying the energies with the LSTM outputs
        weigthed_outputs = torch.mul(rnn_outputs, attn_weights)

        # Reducing dimensions. From (tweet lengt, 2*hidden size) to (2*hidden size) 
        # Summing the outputs after they were multiplied with the energies
        output = torch.sum(weigthed_outputs, -2)

        # Inserting the one output into another linear layer
        output = self.out(output)

        return output
    
    def init_hidden(self):
        return (torch.zeros(self.n_layers*2, 1, self.hidden_size),
                    torch.zeros(self.n_layers*2, 1, self.hidden_size))

with open(os.path.join(currDir,"vocab.pkl"), 'rb') as input:
	vocab = pickle.load(input)	
	
model = torch.load("trainedModel.pt", map_location='cpu')

import sys
import numpy

sentence = sys.argv[1]
sentence = sentence.replace("\\xe2\\x80\\x99", "'")
sentence = sentence.replace("&amp;", "&")
sentence = re.sub(r"http\S+", "", sentence)

if sentence.startswith('b\'') or sentence.startswith('b\"'):
    sentence = sentence[2:]
if sentence.endswith('\'') or sentence.endswith('\"'):
    sentence = sentence[:-1]

sentence = sentence.lstrip(' ')
sentence = re.sub(r"@\S+", "", sentence)
sentence = re.sub(' +', ' ', sentence)   
sentence = sentence.split()             
indexes = vocab.index_document(sentence)

if (len(indexes) == 0):
    maxID = numpy.random.choice((vocab.tag2id[-1], vocab.tag2id[0], vocab.tag2id[1]), p=[0.334, 0.382, 0.284])
else:    
    sentence = torch.tensor(indexes)
    output = model(sentence)
    max = output[0]
    maxID = 0
    if (output[1] > max):
        max = output[1]
        maxID = 1
    if (output[2] > max):
        max = output[2]
        maxID = 2

print (vocab.id2tag[maxID])	