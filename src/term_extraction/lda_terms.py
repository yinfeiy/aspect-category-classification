#! /usr/bin/env python

import os, sys
sys.path.append('../')

import utils.data_helpers as data_helpers
import argparse
from spacy.en import English

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--genre', default='restaurants', help="laptop | restaurants")
args = parser.parse_args()

genre = args.genre
train_data_file = os.path.join(project_path, "data/reviews/review_16_{0}_with_term.train".format(genre))
test_data_file = os.path.join(project_path, "data/reviews/review_16_{0}_with_term.test".format(genre))

if genre == 'laptop':
	model_path = '/Users/yinfei.yang/workspace/nlp/Twitter-LDA/data/results/electronics_no_sw/'
elif genre == 'restaurants':
	model_path = '/Users/yinfei.yang/workspace/nlp/Twitter-LDA/data/results/restaurants_no_sw/'

num_topics = 10

fn_wordmap = model_path + 'uniWordMap.txt'
fn_topics = model_path + 'Topics.txt'
fn_cats = model_path + 'TopicsDistributionOnUsers.txt'

print ('Start loading Twitter-LDA model...')
print ('Loading topic distribution...')
cat_dists = {}
with open(fn_cats) as fin:
    for line in fin:
        ts = line.strip().split()
        cat_name = ts[0].replace('.review', '')
        dists = [ float(x) for x in ts[1:] ]
        cat_dists[cat_name] = dists

print ('Loading vacobulary...')
word2id = {}
id2word = {}
with open(fn_wordmap) as fin:
    for id, word in enumerate(fin.readlines()):
        word = word.strip()
        id2word[id] = word
        word2id[word] = id

print ('Loading word distribution for each topic...')
topics = [{} for i in range(num_topics)]

current_topic_id = -1
with open(fn_topics) as fin:
    for line in fin:
        if line.startswith('Topic'):
            current_topic_id = int(line.strip().replace(':', ' ').split()[1])
            continue
        if current_topic_id >= num_topics:
            break
        print ('Topic :', current_topic_id)
        scores = [float(x) for x in line.strip().split()]
        for id, score in enumerate(scores):
            topics[current_topic_id][id2word[id]] = score

min_probs = []
for i in range(num_topics):
    min_probs.append(min(topics[i].values()))

print ('Load Twitter-LDA model done.')

print ('Start loading spaCy English model...')
en = English()
print ('Load spaCy English done.')

data_helpers.load_data_and_term_labels(train_data_file, test_data_file)
x_text_train, y_train_labels, x_text_test, y_test_labels, labels = \
        data_helpers.load_data_and_term_labels(train_data_file, test_data_file)

for text, text_labels in zip(x_text_train, y_train_labels):

    print text
    doc = en(u'{0}'.format(text))
    probs = {}
    for word in doc:
        word = str(word)
        probs[word] = 0
        for topic_id in range(num_topics):
            probs[word] = max(probs[word], topics[topic_id].get(word, min_probs[topic_id]))

    ncs = []
    for nc in doc.noun_chunks:
        prob = 0
        for word in nc:
            word = str(word)
            prob = max(prob, probs[word])
        ncs.append((str(nc), prob))
    ncs.sort(key=lambda x: x[1], reverse=True)

    for label in text_labels:
        for nc in ncs:
            pass
   # sys.exit(1)
