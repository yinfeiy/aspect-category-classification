#! /usr/bin/env python

import os, sys
sys.path.append('../')

import utils.data_helpers as data_helpers
from gensim.models import Word2Vec
import argparse
import time

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--genre', default='restaurants', help="laptop | restaurants")
args = parser.parse_args()

genre = args.genre
train_data_file = os.path.join(project_path, "data/reviews/review_16_{0}_with_term.train".format(genre))
test_data_file = os.path.join(project_path, "data/reviews/review_16_{0}_with_term.test".format(genre))

if genre == 'laptop':
    word2vec_model = '/Users/yinfei.yang/workspace/nlp/word2vec/models/vectors-reviews-electronics-w.bin'
elif genre == 'restaurants':
    word2vec_model = '/Users/yinfei.yang/workspace/nlp/word2vec/models/vectors-reviews-restaurants.bin'

w2v_model = Word2Vec.load_word2vec_format(word2vec_model, binary=True)

data_helpers.load_data_and_term_labels(train_data_file, test_data_file)
x_text_train, y_train_labels, x_text_test, y_test_labels, labels = \
        data_helpers.load_data_and_term_labels(train_data_file, test_data_file)

for text, labels in zip(x_text_train, y_train_labels):
    words = text.split()

    print '#' * 50
    for label in labels:
        print '#' * 20
        print 'Label: ', label
        aspect_part_1, aspect_part_2 = label[0].lower().split('#')
        #if aspect_part_2 == 'general':
        aspect_name = aspect_part_1
        print 'Aspect: ', aspect_name

        for word in words:
            print word, w2v_model.similarity(aspect_name, word)
        print '#' * 20

    #print w2v_model[word]
    time.sleep(2)
