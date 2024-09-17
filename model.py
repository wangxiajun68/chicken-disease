from preprocessing import EntDataset, load_data, load_word_vectors, sentence_tokenizer, sentence_model
from transformers import BertTokenizerFast, BertModel ,AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import logging
import torch
import numpy as np
import torch.nn as nn

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        
        if Y == 0 or Z == 0:
            f1, precision, recall = 0, 0, 0
        else:
            f1 = 2 * X / (Y + Z)
            precision = X / Y
            recall = X / Z

        return f1, precision, recall

class EfficientGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, external_vector_size, sentence_vector_size, lstm_hidden_size, num_lstm_layers=1, dropout=0.4, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size + external_vector_size + sentence_vector_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.RoPE = RoPE
        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(self.hidden_size, self.lstm_hidden_size, num_layers=self.num_lstm_layers, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(self.lstm_hidden_size * 2, self.ent_type_size * self.inner_dim * 2)

    def forward(self, input_ids, attention_mask, token_type_ids, external_vectors, sentence_embeddings):
        self.device = input_ids.device
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        last_hidden_state = context_outputs[0]
        concat_state = torch.cat((last_hidden_state, external_vectors), dim=-1)
        sentence_embeddings_expanded = sentence_embeddings.unsqueeze(1).expand(-1, concat_state.size(1), -1)
        concat_state = torch.cat((concat_state, sentence_embeddings_expanded), dim=-1)
        lstm_output, _ = self.bilstm(concat_state)
        lstm_output = self.dropout(lstm_output)
        batch_size = lstm_output.size()[0]
        seq_len = lstm_output.size()[1]
        outputs = self.dense(lstm_output)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[...,:self.inner_dim], outputs[...,self.inner_dim:]
        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim**0.5


bert_model_path = 'roberta'
train_cme_path = 'train.json'
eval_cme_path = 'dev.json'
word_vectors_path = 'word_vector.txt'
device = torch.device("cuda:2")
BATCH_SIZE = 32
ENT_CLS_NUM = 10

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)
word_vectors = load_word_vectors(word_vectors_path)

sentence_tokenizer = AutoTokenizer.from_pretrained('sbert-chinese-general-v2')
sentence_model = AutoModel.from_pretrained('sbert-chinese-general-v2')

ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer, word_vectors=word_vectors, sentence_tokenizer=sentence_tokenizer, sentence_model=sentence_model)
ner_loader_train = DataLoader(ner_train, batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True, num_workers=16)
ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer, word_vectors=word_vectors, sentence_tokenizer=sentence_tokenizer, sentence_model=sentence_model)
ner_loader_evl = DataLoader(ner_evl, batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=16)

encoder = BertModel.from_pretrained(bert_model_path)
external_vector_size = next(iter(word_vectors.values())).shape[0]
sentence_vector_size = sentence_model.config.hidden_size
lstm_hidden_size = 256
model = GlobalPointer(encoder, ENT_CLS_NUM, 256, external_vector_size, sentence_vector_size, lstm_hidden_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss

metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
logging.basicConfig(filename='1.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

for eo in range(20):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = batch
        input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device), external_vectors.to(device), sentence_embeddings.to(device)
        logits = model(input_ids, attention_mask, segment_ids, external_vectors, sentence_embeddings)
        loss = loss_fun(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sample_f1 = metrics.get_sample_f1(logits, labels)
        total_loss += loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        print("train_loss:", avg_loss, "\t train_f1:", avg_f1)

    with torch.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Evaluating"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = batch
            input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device), external_vectors.to(device), sentence_embeddings.to(device)
            logits = model(input_ids, attention_mask, segment_ids, external_vectors, sentence_embeddings)
            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = total_f1_ / len(ner_loader_evl)
        avg_precision = total_precision_ / len(ner_loader_evl)
        avg_recall = total_recall_ / len(ner_loader_evl)
        print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1, avg_precision, avg_recall))
        logging.info(f"EPOCH: {eo}\tEVAL_F1: {avg_f1}\tPrecision: {avg_precision}\tRecall: {avg_recall}")

        if avg_f1 > max_f:
            torch.save(model.state_dict(), './outputs/1.pth'.format(eo))
            max_f = avg_f1
        model.train()
