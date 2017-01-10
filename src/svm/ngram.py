import numpy as np

def build_unigram_feature(texts, dict_file='../../dict/restaurant.unigram'):

    # read dict
    feat_words = []
    with open(dict_file) as fin:
        for line in fin:
            ts = line.strip().split()
            key = ts[0]
            feat_words.append(key)

    feat_dict = dict(zip(feat_words, range(len(feat_words))))

    feats = []
    for text in texts:
        feat = [0] * len(feat_words)
        for word in text.strip().split():
            word_idx = feat_dict.get(word, -1)
            if word_idx >= 0:
                feat[word_idx] += 1
        if sum(feat) > 0:
            feat = [f*1.0/sum(feat) for f in feat]
        feats.append(feat)
    feats = np.array(feats)
    return feats

def build_bigram_feature(texts, dict_file='../../dict/restaurant.bigram'):

    # read dict
    feat_words = []
    with open(dict_file) as fin:
        for line in fin:
            ts = line.strip().split()
            key = ts[0]
            feat_words.append(key)

    feat_dict = dict(zip(feat_words, range(len(feat_words))))

    feats = []
    for text in texts:
        feat = [0] * len(feat_words)
        words = text.strip().split()
        for w1, w2 in zip(words[:-1], words[1:]):
            key = '{0}_{1}'.format(w1, w2)
            word_idx = feat_dict.get(key, -1)
            if word_idx >= 0:
                feat[word_idx] += 1
        if sum(feat) > 0:
            feat = [f*1.0/sum(feat) for f in feat]
        feats.append(feat)
    feats = np.array(feats)
    return feats
