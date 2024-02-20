import random
import numpy as np


vocabulary_file='word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)


while True:
    input_term = input("\nEnter three words (EXIT to break): ")
    if input_term.upper() == 'EXIT':
        break
    else:
        np_var = np.array(vectors[input_term])
        euclid_distance = np.linalg.norm(W - np_var, axis = 1)
        sorted_index = np.argsort(euclid_distance)
        low_index = sorted_index[0:3]
        low_val = euclid_distance[low_index]
        print("\n                               Word       Distance\n")
        for i in range(3):
            print("%35s\t\t%f\n" % (ivocab[low_index[i]], euclid_distance[i]))

