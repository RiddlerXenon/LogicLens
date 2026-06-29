import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import evaluate
from sklearn.metrics import classification_report

DATA  = "/home/riddlerxenon/logical-fallacy/data"
MODEL = "xlm-roberta-base"
OUT   = "/data/logiclens/checkpoints-xlmr-v3"

# ── данные: edu + climate ────────────────────────────────────────────
edu_train  = pd.read_csv(f"{DATA}/edu_train.csv")[["source_article", "updated_label"]]
clim_train = pd.read_csv(f"{DATA}/climate_train.csv")[["source_article", "logical_fallacies"]]
clim_train = clim_train.rename(columns={"logical_fallacies": "updated_label"})
valid      = set(edu_train["updated_label"].unique())
clim_train = clim_train[clim_train["updated_label"].isin(valid)]
train_df   = pd.concat([edu_train, clim_train], ignore_index=True).sample(frac=1, random_state=42)

dev_df  = pd.read_csv(f"{DATA}/edu_dev.csv")  [["source_article", "updated_label"]]
test_df = pd.read_csv(f"{DATA}/edu_test.csv") [["source_article", "updated_label"]]

LABELS = sorted(train_df["updated_label"].unique().tolist())
l2id   = {l: i for i, l in enumerate(LABELS)}
id2l   = {i: l for l, i in l2id.items()}
print(f"Классы ({len(LABELS)}):", LABELS)
print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")

for df in [train_df, dev_df, test_df]:
    df["source_article"] = df["source_article"].fillna("").astype(str)
    df["label"] = df["updated_label"].map(l2id)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS), id2label=id2l, label2id=l2id
)

def tokenize(batch):
    return tokenizer(batch["source_article"], truncation=True, max_length=256)

train_ds = Dataset.from_pandas(train_df[["source_article", "label"]]).map(tokenize, batched=True)
dev_ds   = Dataset.from_pandas(dev_df  [["source_article", "label"]]).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test_df [["source_article", "label"]]).map(tokenize, batched=True)

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
    output_dir=OUT,
    num_train_epochs=12,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=0.5,              # стабилизирует grad_norm
    label_smoothing_factor=0.1,     # борьба с переобучением
    eval_strategy="epoch",
    save_strategy="best",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=True,
    lr_scheduler_type="cosine",
    report_to="none",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

print("\n=== TEST RESULTS ===")
results = trainer.evaluate(test_ds)
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

preds_out = trainer.predict(test_ds)
preds     = np.argmax(preds_out.predictions, axis=-1)
print("\n=== PER-CLASS REPORT ===")
print(classification_report(
    test_df["label"].values, preds,
    target_names=[id2l[i] for i in range(len(LABELS))],
    zero_division=0
))

trainer.save_model("/data/logiclens/classifier-xlmr")
tokenizer.save_pretrained("/data/logiclens/classifier-xlmr")
print("\nМодель сохранена в /data/logiclens/classifier-xlmr")
