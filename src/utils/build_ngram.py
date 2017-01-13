import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--genre', help="laptop | restaurants", required=True)
args = parser.parse_args()

input_path = '../../data/reviews/'
genre = args.genre

fn_train = os.path.join(input_path, 'review_16_{0}.train'.format(genre))

# unigram
unigram_dict = {}

# bigram
bigram_dict = {}

with open(fn_train) as fin:
    for line in fin:
        ts = line.strip().split('\t')
        text = ts[0]
        tokens = text.strip().split()

        for token in tokens:
            unigram_dict[token] = unigram_dict.get(token,0) + 1

        for t1, t2 in zip(tokens[:-1], tokens[1:]):
            key = '{0}_{1}'.format(t1, t2)
            bigram_dict[key] = bigram_dict.get(key,0) + 1

uni_words = [word for word in unigram_dict.keys() if unigram_dict[word] >= 3]
bi_words = [word for word in bigram_dict.keys() if bigram_dict[word] >= 3]

ofn_uni = '{0}.unigram'.format(genre)
uni_words.sort(key=lambda x: unigram_dict[x], reverse=True)
with open(ofn_uni, 'w+') as fout:
    for word in uni_words:
        fout.write('{0} {1}\n'.format(word, unigram_dict[word]))

ofn_bi = '{0}.bigram'.format(genre)
bi_words.sort(key=lambda x: bigram_dict[x], reverse=True)
with open(ofn_bi, 'w+') as fout:
    for word in bi_words:
        fout.write('{0} {1}\n'.format(word, bigram_dict[word]))
