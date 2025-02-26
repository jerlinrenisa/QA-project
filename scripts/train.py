# Model training 
import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset (replace 'ag_news' with your dataset)
dataset = load_dataset("ag_news")

# Load tokenizer and model
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))  # Use subset for quick training
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./bert_text_classifier")
tokenizer.save_pretrained("./bert_text_classifier")

print("Training complete! Model saved.")
