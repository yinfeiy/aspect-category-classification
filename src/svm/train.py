#! /usr/bin/env python

import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import utils.data_helpers as data_helpers
from ngram import build_unigram_feature, build_bigram_feature
from word2vec import build_word2vec_feature
from sklearn.svm import SVC, OneClassSVM
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

tf.flags.DEFINE_string("train_data_file", "../../data/reviews/review_16_restaurant.train", "Data source for the training data.")
tf.flags.DEFINE_string("test_data_file", "../../data/reviews/review_16_restaurant.test", "Data source for the testing data.")
tf.flags.DEFINE_string("unigram_dict_file", "../../dict/reviews/restaurant.unigram", "Unigram dictionary file.")
tf.flags.DEFINE_string("bigram_dict_file", "../../dict/reviews/restaurant.bigram", "Bigram dictionary file.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

print("")

# Load data
print ('Loading Data...')
x_train_text, y_train, x_dev_text, y_dev, labels = data_helpers.load_data_and_labels_multi_class(FLAGS.train_data_file, FLAGS.test_data_file, verbose=False)
print ('Loading Data Done.\n')

y_train = np.array(y_train)
y_dev = np.array(y_dev)

# Build n-gram features
print ('Building n-gram features...')
x_train_unigram = build_unigram_feature(x_train_text)
x_dev_unigram = build_unigram_feature(x_dev_text)

x_train_bigram = build_bigram_feature(x_train_text)
x_dev_bigram = build_bigram_feature(x_dev_text)
print ('Building n-gram features done...\n')

print ('Build word2vec feature...')
word2vec_model = '/Users/yinfei.yang/workspace/nlp/word2vec/models/vectors-reviews-restaurants.bin'
x_train_w2v = build_word2vec_feature(x_train_text, word2vec_model)
x_dev_w2v = build_word2vec_feature(x_dev_text, word2vec_model)
print ('Build word2vec feature done...\n')

x_train = np.hstack([x_train_unigram, x_train_bigram, x_train_w2v])
x_dev = np.hstack([x_dev_unigram, x_dev_bigram, x_dev_w2v])

# one v.s. all training for each label
for label_idx, label in enumerate(labels):
    y_train_single = y_train[:, label_idx]
    y_dev_single = y_dev[:, label_idx]

    print ('Processing label {0}'.format(label))
    print ('Number of positive samples {0}/{1}'.format(sum(y_train_single), len(y_train_single)))

    # Building model
    print ('Training models...')
    clf = SVC(kernel='linear', probability=True)
    #clf = OneClassSVM(kernel='linear')
    clf.fit(x_train, y_train_single)
    print ('Training models done...\n')

    print ('Scoreing...')
    y_pred_train = clf.predict_proba(x_train)
    y_pred_dev = clf.predict_proba(x_dev)

    print y_dev_single.shape, y_pred_dev.shape
    precision, recall, thresholds = precision_recall_curve(y_dev_single, y_pred_dev[:,1])
    average_precision = average_precision_score(y_dev_single, y_pred_dev[:,1])

    plt.clf()
    plt.plot(recall, precision, lw=2, color='navy',
                     label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    plt.show()

    #print ('Training Precision : ', metrics.precision_score(y_train_single, y_pred_train))
    #print ('Training Recall : ', metrics.recall_score(y_train_single, y_pred_train))
    #print ('Testing Precision : ', metrics.precision_score(y_dev_single, y_pred_dev))
    #print ('Testing Recall : ', metrics.recall_score(y_dev_single, y_pred_dev))
    print '####\n'

