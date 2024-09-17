from transformers import BertModel, BertTokenizerFast
from model import EfficientGlobalPointer
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Model paths
bert_model_path = 'roberta'  # Your RoBert_large path
save_model_path = 'model_save.pth'
device = torch.device("cuda:1")

# Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder = BertModel.from_pretrained(bert_model_path)
model = EfficientGlobalPointer(encoder, 5, 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:1'))
model.eval()

max_len = 128
ent2id, id2ent = {"bodypart": 0, "type": 1, "disease": 2, "symptom": 3, "drug": 4}, {}
for k, v in ent2id.items():
    id2ent[v] = k

# Function to perform NER prediction
def NER_RELATION(text, tokenizer, ner_model, max_len=128):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])

    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
    scores = model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx": new_span[start][0], "end_idx": new_span[end][-1], "type": id2ent[l]})

    return {"text": text, "entities": entities}

# Function to extract nested entities
def extract_nested_entities(entities):
    nested_entities = []
    entities_sorted = sorted(entities, key=lambda x: (x['start_idx'], x['end_idx']))
    
    for i, entity in enumerate(entities_sorted):
        for j in range(i + 1, len(entities_sorted)):
            if (entities_sorted[j]['start_idx'] >= entity['start_idx'] and 
                entities_sorted[j]['end_idx'] <= entity['end_idx']):
                nested_entities.append(entities_sorted[j])
    
    return nested_entities

# Function to calculate precision, recall, and F1 score
def calculate_prf1(true_entities, pred_entities):
    true_set = set((e['start_idx'], e['end_idx'], e['type']) for e in true_entities)
    pred_set = set((e['start_idx'], e['end_idx'], e['type']) for e in pred_entities)
    
    tp = len(true_set & pred_set)  # True Positives
    fp = len(pred_set - true_set)  # False Positives
    fn = len(true_set - pred_set)  # False Negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Function to process and evaluate predictions
def process_and_evaluate(true_file, pred_file):
    true_nested_entities = []
    pred_nested_entities = []

    with open(true_file, 'r', encoding='utf-8') as true_f, open(pred_file, 'r', encoding='utf-8') as pred_f:
        for true_line, pred_line in zip(true_f, pred_f):
            true_data = json.loads(true_line.strip())
            pred_data = json.loads(pred_line.strip())

            # Extract nested entities
            true_nested = extract_nested_entities(true_data['entities'])
            pred_nested = extract_nested_entities(pred_data['entities'])

            true_nested_entities.extend(true_nested)
            pred_nested_entities.extend(pred_nested)

    precision, recall, f1 = calculate_prf1(true_nested_entities, pred_nested_entities)
    return precision, recall, f1

if __name__ == '__main__':
    # Prediction phase
    all_ = []
    for d in tqdm(json.load(open('test.json'))):
        all_.append(NER_RELATION(d["text"], tokenizer=tokenizer, ner_model=model))

    pred_file = 'pred.json'
    true_file = 'test.json'

    # Save prediction results
    json.dump(all_, open(pred_file, 'w'), indent=4, ensure_ascii=False)

    # Evaluate predictions
    precision, recall, f1 = process_and_evaluate(true_file, pred_file)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
