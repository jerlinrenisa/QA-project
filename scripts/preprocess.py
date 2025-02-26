# Data preprocessing 
import re
import string
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

# Load tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text.strip()

# Function to tokenize text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load dataset (Replace with actual dataset)
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)  # Assuming a CSV file with "text" and "label" columns
    df["text"] = df["text"].apply(clean_text)  # Apply text cleaning
    dataset = Dataset.from_pandas(df)  # Convert to Hugging Face dataset format
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Run preprocessing
if __name__ == "__main__":
    dataset_path = "data.csv"  # Change to your dataset path
    processed_data = load_and_preprocess_data(dataset_path)
    print("Preprocessing complete!")
    print(processed_data)
