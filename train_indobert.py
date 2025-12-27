# ============================================================
# TRAIN INDOBERT - Fine-tuning untuk Sentimen MBG
# ============================================================
# Jalankan dengan: python train_indobert.py
# Disarankan pakai GPU untuk mempercepat training
# ============================================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "indobenchmark/indobert-base-p1"  # IndoBERT base model
OUTPUT_DIR = "./models/indobert"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# ============================================================
# CHECK GPU
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️ WARNING: Training tanpa GPU akan sangat lambat!")
    print("   Disarankan pakai Google Colab atau PC dengan GPU NVIDIA")

# ============================================================
# LOAD DATA
# ============================================================
print("\n📂 Loading dataset...")
df = pd.read_csv('data/dataset.csv')

# Ambil kolom yang diperlukan
texts = df['cleaned_comment'].fillna('').tolist()
labels_raw = df['sentimen_relabel'].tolist()

# Mapping label ke angka
label_map = {'negatif': 0, 'netral': 1, 'positif': 2}
labels = [label_map[l] for l in labels_raw]

print(f"   Total data: {len(texts)}")
print(f"   Distribusi: {pd.Series(labels_raw).value_counts().to_dict()}")

# ============================================================
# SPLIT DATA
# ============================================================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\n📊 Split data:")
print(f"   Training: {len(train_texts)}")
print(f"   Validation: {len(val_texts)}")

# ============================================================
# TOKENIZER
# ============================================================
print("\n📝 Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# DATASET CLASS
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

# ============================================================
# LOAD MODEL
# ============================================================
print("\n🤖 Loading IndoBERT model...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,  # negatif, netral, positif
    id2label={0: 'negatif', 1: 'netral', 2: 'positif'},
    label2id={'negatif': 0, 'netral': 1, 'positif': 2}
)

# ============================================================
# TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",  # Ganti dari evaluation_strategy (deprecated)
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"
)

# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# ============================================================
# TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ============================================================
# TRAINING
# ============================================================
print("\n" + "="*60)
print("🚀 MEMULAI TRAINING INDOBERT")
print("="*60)
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Max length: {MAX_LENGTH}")
print("="*60 + "\n")

trainer.train()

# ============================================================
# SAVE MODEL
# ============================================================
print("\n💾 Menyimpan model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Model berhasil disimpan ke: {OUTPUT_DIR}")

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "="*60)
print("📊 EVALUASI MODEL")
print("="*60)

# Predict on validation set
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)

# Classification report
label_names = ['negatif', 'netral', 'positif']
print(classification_report(val_labels, preds, target_names=label_names))

print("\n✅ Training selesai!")
print(f"   Model tersimpan di: {OUTPUT_DIR}")
print("   Sekarang kamu bisa menjalankan Streamlit dan pilih IndoBERT")
