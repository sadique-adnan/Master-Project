#! /usr/bin/env python

import numpy as np
import pickle
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('wiki.en.vec')
vocab = model.vocab
embeddings = np.array([model.word_vec(k) for k in vocab.keys()])

with open('fasttext_vocab_en.dat', 'wb') as fw:
    pickle.dump(vocab, fw, protocol=pickle.HIGHEST_PROTOCOL)

np.save('fasttext_embedding_en.npy', embeddings)
