# BERT for Named Entity Recognition (CoNLL-2003)

## Overview
Fine-tuned a pre-trained BERT model using the Hugging Face `transformers` library for Token Classification (Named Entity Recognition) on the CoNLL-2003 dataset. This project demonstrates end-to-end NLP fine-tuning, custom metric evaluation, and model checkpointing in PyTorch.

## Results
- **Model**: `bert-base-cased`
- **Dataset**: CoNLL-2003 (via Hugging Face Hub)
- **Evaluation Metric**: Entity-level F1-score (via `seqeval`)
- **Test Set Performance**: 
  - **F1-Score: 0.9084**
  - Precision: 0.9021
  - Recall: 0.9148
  - Accuracy: 0.9822

## Approach & Key Technical Details
1. **Data Preprocessing**: Used `BertTokenizerFast` with `is_split_into_words=True` to properly align NER tags with subword tokens. Handled special tokens and padding by assigning a label of `-100` to ignore them during loss calculation.
2. **Model Architecture**: Loaded `AutoModelForTokenClassification` with custom `id2label` and `label2id` mappings for the 9 CoNLL tags (PER, ORG, LOC, MISC, O).
3. **Training**: Fine-tuned using the Hugging Face `Trainer` API with the following hyperparameters:
   - Epochs: 3
   - Batch Size: 16 (Train & Eval)
   - Learning Rate: 2e-5
   - Weight Decay: 0.01
   - Max Sequence Length: 128
4. **Evaluation**: Implemented a custom `compute_metrics` function using the `seqeval` library to calculate true entity-level precision, recall, and F1, rather than relying on token-level accuracy.

## How to Run
1. Clone this repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
