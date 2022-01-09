from utils import *
from collections import Counter
vocab = Counter()

process_docs_to_vocab('txt_sentoken/neg', vocab)
process_docs_to_vocab('txt_sentoken/pos', vocab)

print(len(vocab))

min_occurrence = 5
tokens = [k for k,c in vocab.items() if c >= min_occurrence]

print(tokens)
save_list(tokens, 'vocab.txt')