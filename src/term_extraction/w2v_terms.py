#! /usr/bin/env python

import os, sys
sys.path.append('../')

import utils.data_helpers as data_helpers
from gensim.models import Word2Vec
import argparse
import time
from spacy.en import English

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--genre', default='restaurants', help="laptop | restaurants")
args = parser.parse_args()

genre = args.genre
train_data_file = os.path.join(project_path, "data/reviews/review_16_{0}_with_term.train".format(genre))
test_data_file = os.path.join(project_path, "data/reviews/review_16_{0}_with_term.test".format(genre))

if genre == 'laptop':
    word2vec_model = '/Users/yinfei.yang/workspace/nlp/word2vec/models/vectors-reviews-electronics.bin'
elif genre == 'restaurants':
    word2vec_model = '/Users/yinfei.yang/workspace/nlp/word2vec/models/vectors-reviews-restaurants.bin'

w2v_model = Word2Vec.load_word2vec_format(word2vec_model, binary=True)

data_helpers.load_data_and_term_labels(train_data_file, test_data_file)
x_text_train, y_train_labels, x_text_test, y_test_labels, labels = \
        data_helpers.load_data_and_term_labels(train_data_file, test_data_file)

en = English()
cd = 0
total = 0

for text, labels in zip(x_text_test, y_test_labels):

    doc = en(u'{0}'.format(text))
    noun_chunks = [str(nc) for nc in doc.noun_chunks]

    #words = text.split()

    for label in labels:
        total += 1

        label_term = label[1]
        flag = False
        for nc in noun_chunks:
            if nc.find(label_term) >=0 or label_term=='null':
                cd +=1
                flag=True
                break

        if not flag:
            print '##'
            print text
            print noun_chunks, label
            print '##'
        print cd, total

        #for word in words:
        #    print word, w2v_model.similarity(aspect_name, word)
    #print w2v_model[word]
