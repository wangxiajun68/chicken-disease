import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta')
model = RobertaModel.from_pretrained('roberta')

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    return corpus

def get_char_embeddings(corpus):
    char_embeddings = {}
    chars = list(set(corpus))
    
    for char in chars:
        inputs = tokenizer(char, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        char_embeddings[char] = embedding[0]
    
    return char_embeddings

def load_word_vectors(file_path):
    vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=float)
            vectors[word] = vector
    return vectors

def load_word_frequencies(file_path):
    word_freq = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            frequency = int(tokens[1])
            word_freq[word] = frequency
    return word_freq

def split_heads(x, num_heads):
    depth = x.shape[-1] // num_heads
    return np.reshape(x, (x.shape[0], num_heads, x.shape[1], depth))

def multihead_attention(query, keys, values, freqs, num_heads=8):
    query_heads = split_heads(np.expand_dims(query, axis=0), num_heads)
    keys_heads = split_heads(keys, num_heads)
    values_heads = split_heads(values, num_heads)
    head_outputs = []
    for i in range(num_heads):
        query_i = query_heads[0, i, 0]
        keys_i = keys_heads[:, i, :]
        values_i = values_heads[:, i, :]
        scores = np.dot(keys_i, query_i)
        scores = scores * freqs
        attention_weights = np.exp(scores) / np.sum(np.exp(scores))
        weighted_sum = np.sum(attention_weights[:, np.newaxis] * values_i, axis=0)
        head_outputs.append(weighted_sum)
    final_output = np.concatenate(head_outputs, axis=-1)
    return final_output

corpus = load_corpus("corpus.txt")
char_vectors = get_char_embeddings(corpus)

word_vectors = load_word_vectors("word_vectors.txt")
word_frequencies = load_word_frequencies("word_frequencies.txt")

char_to_word_vectors = {}
char_to_word_freq = {}
for word, vector in word_vectors.items():
    for char in word:
        if char not in char_to_word_vectors:
            char_to_word_vectors[char] = []
            char_to_word_freq[char] = []
        char_to_word_vectors[char].append(vector)
        if word in word_frequencies:
            char_to_word_freq[char].append(word_frequencies[word])
        else:
            char_to_word_freq[char].append(1)

final_char_vectors = {}
for char, vectors in char_to_word_vectors.items():
    if char in char_vectors:
        char_vector = char_vectors[char]
        vectors = np.array(vectors)
        freqs = np.array(char_to_word_freq[char])
        freqs = freqs / np.max(freqs)
        final_vector = multihead_attention(char_vector, vectors, vectors, freqs, num_heads=8)
        final_char_vectors[char] = final_vector

with open('char_vectors.txt', 'w', encoding='utf-8') as f:
    for char, vector in final_char_vectors.items():
        vector_str = ' '.join(map(str, vector))
        f.write(f"{char} {vector_str}\n")
