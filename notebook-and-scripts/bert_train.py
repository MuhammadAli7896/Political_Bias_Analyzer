import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')
nltk.download('punkt')

import pandas as pd
import nlpaug.augmenter.word as naw
from transformers import pipeline, logging, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import random
import gc

device = 0 if torch.cuda.is_available() else -1
print("✅ Device:", "GPU - " + torch.cuda.get_device_name(0) if device == 0 else "CPU")

logging.set_verbosity_info()

df = pd.read_csv("political_bias_dataset.csv")
print(f"Loaded {len(df)} rows from political_bias_dataset.csv")

print("\n⏳ Downloading and initializing the paraphraser model...")
model_name = "google/flan-t5-small"

AutoTokenizer.from_pretrained(model_name)
AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ Model cached successfully.")

paraphraser = pipeline("text2text-generation", model=model_name, device=device)

syn_aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=3)
print("✅ Augmenters ready.\n")

def batch_paraphrase(texts, batch_size=16):
    """Generate paraphrases in batches for efficiency"""
    paraphrased = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        prompts = [f"Paraphrase this: {t}" for t in batch]
        try:
            outputs = paraphraser(
                prompts,
                max_new_tokens=64,
                num_return_sequences=1
            )
            paraphrased.extend([o['generated_text'] for o in outputs])
        except Exception as e:
            print(f"⚠️ Paraphrasing batch {i} failed:", e)
            paraphrased.extend(batch)  
        torch.cuda.empty_cache()
        gc.collect()
    return paraphrased

augmented_rows = []

print("\n🚀 Starting augmentation process...")
for subtype, group in tqdm(df.groupby("subtype"), desc="Augmenting by subtype"):
    texts = group["text"].tolist()
    labels = group["subtype"].tolist()

    paras = batch_paraphrase(texts)
    syns = [syn_aug.augment(t) for t in texts]

    for orig, para, syn, label in zip(texts, paras, syns, labels):
        augmented_rows.append((orig, label))
        augmented_rows.append((para, label))
        augmented_rows.append((syn, label))

aug_df = pd.DataFrame(augmented_rows, columns=["text", "subtype"])
aug_df.to_csv("political_bias_dataset_nlp_aug.csv", index=False)
print(f"\n✅ Augmentation complete! Generated {len(aug_df)} rows.")
print("💾 Saved as political_bias_dataset_nlp_aug.csv")

!pip install scikit-learn

import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import evaluate
from sklearn.model_selection import train_test_split
import torch
import numpy as np

df = pd.read_csv("dataset.csv")  

label2id = {label: i for i, label in enumerate(df["bias_rating"].unique())}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["bias_rating"].map(label2id)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir="./roberta_bias",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("roberta_bias_model")
tokenizer.save_pretrained("roberta_bias_model")







df = pd.read_csv("dataset.csv")  

label2id = {label: i for i, label in enumerate(df["type_of_biasness"].unique())}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["type_of_biasness"].map(label2id)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir="./roberta_subtype",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("roberta_subtype_model")
tokenizer.save_pretrained("roberta_subtype_model")

!pip install torch transformers scikit-learn

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("final_validated.csv")
df.head()

df = pd.read_csv("final_validated.csv")

df["text"] = df["text"].astype(str).fillna("")

df = df[df["text"].str.strip() != ""]

df = df.reset_index(drop=True)

unique_labels = df['type_of_biasness'].unique()
print("Unique Labels:")
print(unique_labels)

label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['type_of_biasness'])

texts = df['text'].tolist()
labels = df['Label'].tolist()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizerA
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def predict_class(text, model, tokenizer, max_length=128, device='cpu'):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return preds.item()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm



df = pd.read_csv("dataset.csv")

label2id = {label: i for i, label in enumerate(df["bias_rating"].unique())}
df["label"] = df["bias_rating"].map(label2id)

texts = df["text"].tolist()
labels = df["label"].tolist()

print("Label to ID mapping:")
for label, idx in label2id.items():
    print(f"{idx}: {label}")


id2label = {v: k for k, v in label2id.items()}
print("\nID to Label mapping:")
for idx, label in id2label.items():
    print(f"{idx}: {label}")

import json
with open("label_mapping.json", "w") as f:
    json.dump(id2label, f)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

model = BERTClassifier('bert-base-uncased', num_classes=7)  
model.load_state_dict(torch.load("bert_type_classifier.pt"))
model.to(device)

df_new = pd.read_csv("news_articles.csv")  
label2id = {label: i for i, label in enumerate(df_new["bias_rating"].unique())}
df_new["label"] = df_new["bias_rating"].map(label2id)

texts_new = df_new["text"].tolist()
labels_new = df_new["label"].tolist()

train_texts_new, val_texts_new, train_labels_new, val_labels_new = train_test_split(
    texts_new, labels_new, test_size=0.2, random_state=42, stratify=labels_new
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
train_dataset_new = TextClassificationDataset(train_texts_new, train_labels_new, tokenizer, max_length)
val_dataset_new = TextClassificationDataset(val_texts_new, val_labels_new, tokenizer, max_length)

train_dataloader_new = DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True)
val_dataloader_new = DataLoader(val_dataset_new, batch_size=batch_size)

label2id

optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_dataloader_new) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader_new, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader_new)

    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_dataloader_new:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={acc:.4f}, Val F1={f1:.4f}")

torch.save(model.state_dict(), "bert_type_classifier.pt")

