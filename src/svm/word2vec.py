from gensim.models import Word2Vec
import numpy as np

def build_word2vec_feature(texts, fn_model='/Users/yinfei.yang/workspace/nlp/word2vec/models/GoogleNews-vectors-negative300.bin', binary=True):
    model = Word2Vec.load_word2vec_format(fn_model, binary=binary)

    feats = []
    for text in texts:
        feat = np.zeros(300)
        words = text.strip().split()
        for word in words:
            try:
                vec = model[word]
                feat += vec
            except:
                pass
        feat /= len(words)
        feats.append(feat)
    feats = np.array(feats)
    return feats

