#! /usr/bin/env python

import sys
sys.path.append('../')

import argparse
import numpy as np
import utils.data_helpers as data_helpers
from ngram import build_unigram_feature, build_bigram_feature
from word2vec import build_word2vec_feature
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.externals import joblib

import shutil, os

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--genre', default='laptop', help="laptop | restaurant")
parser.add_argument('-p', '--pr_curve', default=False, help="plot precision-recall curve",  action='store_true')
parser.add_argument('-o', '--output_path', default='', help="save trained model")
args = parser.parse_args()

genre = args.genre
plot_pr_curve = args.pr_curve
output_path = args.output_path

if plot_pr_curve:
    import matplotlib.pyplot as plt

train_data_file = os.path.join(project_path, "data/reviews/review_16_{0}.train".format(genre))
test_data_file = os.path.join(project_path, "data/reviews/review_16_{0}.test".format(genre))
unigram_dict_file = os.path.join(project_path, "dict/{0}.unigram".format(genre))
bigram_dict_file = os.path.join(project_path, "dict/{0}.bigram".format(genre))

if genre == 'laptop':
    word2vec_model = '/Users/yinfei.yang/workspace/nlp/word2vec/models/vectors-reviews-electronics-w.bin'
elif genre == 'restaurant':
    word2vec_model = '/Users/yinfei.yang/workspace/nlp/word2vec/models/vectors-reviews-restaurants.bin'
else:
    print 'Error, only support laptop or restaurant'
    sys.exit(1)

# Load data
print ('Loading Data...')
x_train_text, y_train, x_dev_text, y_dev, labels = data_helpers.load_data_and_labels_multi_class(train_data_file, test_data_file, verbose=False)
print ('Loading Data Done.\n')

y_train = np.array(y_train)
y_dev = np.array(y_dev)

# Build n-gram features
print ('Building n-gram features...')
x_train_unigram = build_unigram_feature(x_train_text, dict_file=unigram_dict_file)
x_dev_unigram = build_unigram_feature(x_dev_text, dict_file=unigram_dict_file)

x_train_bigram = build_bigram_feature(x_train_text, dict_file=bigram_dict_file)
x_dev_bigram = build_bigram_feature(x_dev_text, dict_file=bigram_dict_file)
print ('Building n-gram features done...\n')

print ('Build word2vec feature...')
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

    if sum(y_train_single) < 20:
        print ('Number of training samples is too small, stop')
        break

    # Building model
    print ('Training models...')
    clf = SVC(kernel='linear', probability=True)
    clf.fit(x_train, y_train_single)
    print ('Training models done...\n')

    print ('Scoreing...')
    y_pred_train = clf.predict_proba(x_train)
    y_pred_dev = clf.predict_proba(x_dev)

    precision, recall, thresholds = precision_recall_curve(y_dev_single, y_pred_dev[:,1])
    average_precision = average_precision_score(y_dev_single, y_pred_dev[:,1])

    if plot_pr_curve:
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

    y_pred_train = [1 if y[1]>=0.5 else 0 for y in y_pred_train]
    y_pred_dev = [1 if y[1]>=0.5 else 0 for y in y_pred_dev]

    print ('Training Precision : ', metrics.precision_score(y_train_single, y_pred_train))
    print ('Training Recall : ', metrics.recall_score(y_train_single, y_pred_train))
    print ('Testing Precision : ', metrics.precision_score(y_dev_single, y_pred_dev))
    print ('Testing Recall : ', metrics.recall_score(y_dev_single, y_pred_dev))
    print '####\n'

    if output_path:
        model_path = os.path.join(output_path, label)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            shutil.rmtree(model_path)

        joblib.dump(clf, os.path.join(model_path, '{0}.pkl'.format(label)))
