# MFGFF-BiLSTM-EGP: A Chinese Nested Named Entity Recognition Model for Chicken Disease Based on Multiple Fine-Grained Feature Fusion and Efficient Global Pointer


## Required Data

### 1. Training and Evaluation Data
- The datasets should be in **JSON format** with the following files:
  - `train.json`: Training dataset.
  - `dev.json`: Evaluation dataset for model validation during training.
  - `test.json`: Test dataset for final model evaluation.

Each dataset file should have the following structure:
```json
{
  "text": "The patient's condition worsened...",
  "entities": [
    {
      "start_idx": 10,
      "end_idx": 15,
      "type": "disease"
    }
  ]
}
```

### 2. Pre-trained model
RoBERTa https://github.com/brightmart/roberta_zh
SBERT https://huggingface.co/DMetaSoul/sbert-chinese-general-v1

### 3. word vector
Train word vectors using glove https://github.com/stanfordnlp/glove
Tencent AI Word Vector https://ai.tencent.com/ailab/nlp/en/index.html

## Runtime Environment
#### python                   3.8
#### transformers             4.35.0
#### torch                    2.0.1
#### tqdm                     4.66.1
#### numpy                    1.24.4


## file
#### main.py: The main script to train the NER model. It loads the dataset, tokenizers, and initializes the training loop.
#### model.py: Contains the implementation of the EfficientGlobalPointer model and the custom loss function.
#### evaluate.py: Script for evaluating the trained model on test data, calculating metrics like precision, recall, and F1-score.
#### preprocessing.py: Provides data loading utilities and functions for computing sentence embeddings.
###3 word_char.py: Contains functions for processing word and character embeddings using pre-trained RoBERTa models.
