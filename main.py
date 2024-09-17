import torch
from transformers import BertModel, BertTokenizerFast, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from model import EfficientGlobalPointer, MetricsCalculator, loss_fun
from preprocessing import EntDataset, load_data, load_word_vectors
from word_char import get_char_embeddings

# Paths for necessary files
train_data_path = 'train.json'
eval_data_path = 'dev.json'
word_vectors_path = 'word_vectors.txt'
char_corpus_path = 'corpus.txt'

# Load tokenizer and model
bert_model_path = 'roberta'
device = torch.device("cuda:1")

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)
word_vectors = load_word_vectors(word_vectors_path)
char_embeddings = get_char_embeddings(char_corpus_path)

# Sentence tokenizer and model for sentence embeddings
sentence_tokenizer = AutoTokenizer.from_pretrained('sbert-chinese-general-v2')
sentence_model = AutoModel.from_pretrained('sbert-chinese-general-v2')

# Prepare dataset
train_dataset = EntDataset(load_data(train_data_path), tokenizer, word_vectors, sentence_tokenizer, sentence_model)
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=train_dataset.collate, shuffle=True, num_workers=16)
eval_dataset = EntDataset(load_data(eval_data_path), tokenizer, word_vectors, sentence_tokenizer, sentence_model)
eval_loader = DataLoader(eval_dataset, batch_size=32, collate_fn=eval_dataset.collate, shuffle=False, num_workers=16)

encoder = BertModel.from_pretrained(bert_model_path)
external_vector_size = next(iter(word_vectors.values())).shape[0]
sentence_vector_size = sentence_model.config.hidden_size
lstm_hidden_size = 256

model = EfficientGlobalPointer(encoder, 5, 64, external_vector_size, sentence_vector_size, lstm_hidden_size, dropout=0.4).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

total_steps = len(train_loader) * 20
warmup_steps = int(total_steps * 0.1)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))

metrics = MetricsCalculator()
max_f1 = 0.0

for epoch in range(20):
    model.train()
    total_loss = 0.0
    for idx, batch in enumerate(train_loader):
        raw_text_list, input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = batch
        input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = (
            input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device), 
            external_vectors.to(device), sentence_embeddings.to(device)
        )
        
        logits = model(input_ids, attention_mask, segment_ids, external_vectors, sentence_embeddings)
        loss = loss_fun(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Loss: {avg_loss}")

    model.eval()
    total_f1 = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            raw_text_list, input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = batch
            input_ids, attention_mask, segment_ids, labels, external_vectors, sentence_embeddings = (
                input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device), 
                external_vectors.to(device), sentence_embeddings.to(device)
            )
            
            logits = model(input_ids, attention_mask, segment_ids, external_vectors, sentence_embeddings)
            f1, precision, recall = metrics.get_evaluate_fpr(logits, labels)
            total_f1 += f1

    avg_f1 = total_f1 / len(eval_loader)
    print(f"Epoch {epoch} - F1 Score: {avg_f1}")

    if avg_f1 > max_f1:
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        max_f1 = avg_f1
