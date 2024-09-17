import json
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModel

max_len = 128
ent2id = {"bodypart": 0, "type": 1, "disease": 2, "symptom": 3, "drug": 4}
id2ent = {v: k for k, v in ent2id.items()}

def load_word_vectors(filepath):
    word_vectors = {}
    with open(filepath, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

def load_data(path):
    D = []
    for d in json.load(open(path)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, ent2id[label]))
    return D

def compute_sentence_embedding(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).squeeze().numpy()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 加载句向量模型
sentence_tokenizer = AutoTokenizer.from_pretrained('/data/SBERT/sbert-chinese-general-v2')
sentence_model = AutoModel.from_pretrained('/data/SBERT/sbert-chinese-general-v2')

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, word_vectors, sentence_tokenizer, sentence_model, istrain=True):
        self.data = data
        self.tokenizer = tokenizer
        self.word_vectors = word_vectors
        self.sentence_tokenizer = sentence_tokenizer
        self.sentence_model = sentence_model
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item[0]
        token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=max_len, truncation=True)["offset_mapping"]
        start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}

        encoder_txt = self.tokenizer.encode_plus(text, max_length=max_len, truncation=True)
        input_ids = encoder_txt["input_ids"]
        token_type_ids = encoder_txt["token_type_ids"]
        attention_mask = encoder_txt["attention_mask"]

        external_vectors = []
        for token_id in input_ids:
            token = self.tokenizer.decode([token_id])
            if token in self.word_vectors:
                external_vectors.append(self.word_vectors[token])
            else:
                external_vectors.append(np.zeros_like(next(iter(self.word_vectors.values()))))

        external_vectors = np.array(external_vectors)
        sentence_embedding = compute_sentence_embedding(text, self.sentence_tokenizer, self.sentence_model)

        return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, external_vectors, sentence_embedding

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids, batch_external_vectors, batch_sentence_embeddings = [], [], [], [], [], [], []
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask, external_vectors, sentence_embedding = self.encoder(item)

            labels = np.zeros((len(ent2id), max_len, max_len))
            for start, end, label in item[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[label, start, end] = 1
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
            batch_external_vectors.append(external_vectors)
            batch_sentence_embeddings.append(sentence_embedding)

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_external_vectors = torch.tensor(self.sequence_padding(batch_external_vectors, seq_dims=2)).float()
        
        batch_sentence_embeddings = np.array(batch_sentence_embeddings)
        batch_sentence_embeddings = torch.tensor(batch_sentence_embeddings).float()

        return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels, batch_external_vectors, batch_sentence_embeddings


    def __getitem__(self, index):
        item = self.data[index]
        return item
