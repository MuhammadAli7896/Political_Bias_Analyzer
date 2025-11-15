import torch
import torch.nn as nn
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        return self.classifier(dropped)

# ======================
# 2️⃣ Load Model and Tokenizer
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'bert-base-uncased'
num_classes = 3  # Center, Right, Left

model = BERTClassifier(model_name, num_classes)

# Load previous 7-class weights but skip classifier layer
checkpoint = torch.load("bert_weights/bert_type_classifier.pt", map_location=device)
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in checkpoint.items() if 'classifier' not in k}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)
model.to(device)
model.eval()

print("✅ Loaded BERT weights successfully (classifier reinitialized for 3 classes).")

tokenizer = BertTokenizer.from_pretrained(model_name)

# ======================
# 3️⃣ Load Dataset
# ======================
df = pd.read_csv("dataset.csv")   # must contain columns: text, bias_rating
label2id = {'Center': 0, 'Right': 1, 'Left': 2}
id2label = {v: k for k, v in label2id.items()}

texts = df["text"].tolist()
true_labels = [label2id[l] for l in df["bias_rating"].tolist()]

# ======================
# 4️⃣ Predict in Batches
# ======================
all_preds = []

for i in range(0, len(texts), 16):
    batch_texts = texts[i:i+16]
    encodings = tokenizer(
        batch_texts,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)

# ======================
# 5️⃣ Evaluation
# ======================
acc = accuracy_score(true_labels, all_preds)
f1 = f1_score(true_labels, all_preds, average='weighted')
cm = confusion_matrix(true_labels, all_preds)
report = classification_report(true_labels, all_preds, target_names=label2id.keys())

print("\n📊 Evaluation Results")
print("---------------------")
print(f"Accuracy: {acc:.4f}")
print(f"Weighted F1 Score: {f1:.4f}\n")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
