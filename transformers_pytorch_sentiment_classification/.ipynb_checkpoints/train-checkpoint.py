# Training a Transformer model for sentiment classification
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW,  get_scheduler
from tqdm.auto import tqdm

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device  ::  {}".format(device))

checkpoint = 'distilbert-base-uncased'
print("Checkpoint  ::  {}".format(checkpoint))

## Loading Tokenizer
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

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

train_dataloader = DataLoader(
                        tokenized_dataset["train"],
                        shuffle=True,
                        batch_size=16,
                        collate_fn=data_collator
                    )

## Model Training
print("Model Training")

### Loading Model
print("Loading Model")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
model.to(device)
print(model)

### Initialising Optimizer
print("Initialising Optimizer")
optimizer = AdamW(model.parameters(), lr=3e-5)

### Initialising Scheduler
print("Initialising Scheduler")
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print("Number of training steps  ::  {}".format(num_training_steps))

### Training Loop
print("Initialising Training Loop")
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = output.loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

## Saving the model
out_dir = './../../model/fin_sentiment_distilbert/'
print("Saving the model at {}".format(out_dir))
model.save_pretrained(out_dir)