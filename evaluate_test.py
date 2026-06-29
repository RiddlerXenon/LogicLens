import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import evaluate

DATA      = "/home/riddlerxenon/logical-fallacy/data"
MODEL_DIR = "/data/logiclens/classifier"

test_df = pd.read_csv(f"{DATA}/edu_test.csv")[["source_article", "updated_label"]]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

LABELS = sorted(test_df["updated_label"].unique().tolist())
l2id   = {l: i for i, l in enumerate(LABELS)}
test_df["label"] = test_df["updated_label"].map(l2id)

test_ds = Dataset.from_pandas(test_df[["source_article", "label"]])
test_ds = test_ds.map(
    lambda b: tokenizer(b["source_article"], truncation=True, max_length=256),
    batched=True
)

f1_metric  = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
    }

args = TrainingArguments(
    output_dir="/tmp/eval_tmp",
    per_device_eval_batch_size=64,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    eval_dataset=test_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print("\n=== TEST RESULTS (distilbert baseline) ===")
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# Посмотрим на ошибки по классам
from sklearn.metrics import classification_report
preds_out = trainer.predict(test_ds)
preds     = np.argmax(preds_out.predictions, axis=-1)
id2l      = {i: l for l, i in l2id.items()}
print("\n=== PER-CLASS REPORT ===")
print(classification_report(
    test_df["label"].values, preds,
    target_names=[id2l[i] for i in range(len(LABELS))]
))
