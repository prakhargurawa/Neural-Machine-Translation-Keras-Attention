# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 05:02:51 2021

@author: prakh
"""
# Import libraries
import string
from pickle import dump
from numpy import array

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)
    
def preprocess_pairs(lines):
    cleaned = list()
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # remove extra spaces
            line = line.strip()
            # convert to lowercase
            line = line.lower()
            # remove quotes
            line = line.replace('"', '');line=line.replace("'", '')
            # remove punctuation
            exclude = set(string.punctuation)
            line = ''.join(ch for ch in line if ch not in exclude)
            # remove numbers
            line = ''.join([ch for ch in line if not ch.isdigit()])
            clean_pair.append(line)
        cleaned.append(clean_pair)
    return array(cleaned)
            

# load dataset
filename = 'Data\deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pair = preprocess_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pair, 'Temp\english-german.pkl')
# spot check
for i in range(100):
	print('[%s] => [%s]' % (clean_pair[i,0], clean_pair[i,1]))
    