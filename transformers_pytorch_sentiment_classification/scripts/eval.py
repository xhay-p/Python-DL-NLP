# Training a Transformer model for sentiment classification
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device  ::  {}".format(device))

model = 'distilbert-base-uncased'
print("Checkpoint  ::  {}".format(model))

## Loading Tokenizer
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model)

## Dataset Processing
print("Dataset Processing")
### Dataset Loading
print("Dataset Loading")
raw_dataset = load_dataset('financial_phrasebank', 'sentences_allagree')
raw_dataset = raw_dataset['train'].train_test_split(test_size=0.2, stratify_by_column="label")

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

### Initialising Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

### Preparing data for training
print("Preparing data for training")
tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")
print(tokenized_dataset["train"].column_names)

test_dataloader = DataLoader(
                        tokenized_dataset["test"],
                        batch_size=16,
                        collate_fn=data_collator
                    )

## Model EValuation
print("Model Evaluation")
### Loading Model
checkpoint = './../../model/fin_sentiment_distilbert/'
print("Loading Model from {}".format(checkpoint))
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.to(device)
print(model)

### Evaluation Loop
print("Initialising Evaluation Loop")
model.eval()

load_accuracy = load_metric("accuracy")
load_f1 = load_metric("f1")

model.eval()
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    load_accuracy.add_batch(predictions=predictions, references=batch["labels"])
    load_f1.add_batch(predictions=predictions, references=batch["labels"])

print("Accuracy  ::  {}".format(load_accuracy.compute()))
print("F1-Score  ::  {}".format(load_f1.compute(average='macro')))