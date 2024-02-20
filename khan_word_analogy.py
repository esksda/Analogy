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
print("Give dash(-) in between inputs as a-b-c")


# Searching starts now
while True:
    inp = input('\nEnter three words using dash(-) like a-s-d (EXIT to break): ').lower()
    if inp.upper() == 'EXIT':
        break
    else :
        new_inp = inp.split("-")
        first = np.array(vectors[new_inp[0]])
        second = np.array(vectors[new_inp[1]])
        new_sub = second - first
        third = np.array(vectors[new_inp[2]])
        fourth = third + new_sub
        new_euclid = np.linalg.norm(W - fourth, axis = 1)
        new_sort = np.argsort(new_euclid)
        new_low_index = new_sort[0:5]
        new_low_val = new_euclid[new_low_index]
        final = [ivocab[x] for x in new_low_index if ivocab[x] != new_inp[0] and ivocab[x] != new_inp[1] and ivocab[x] != new_inp[2]]
        for i in range(2):
            print(final[i])

        











