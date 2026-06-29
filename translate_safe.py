import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from collections import Counter
import json

DATA  = "/home/riddlerxenon/logical-fallacy/data"
MODEL = "/data/models/qwen3-8b"
OUT   = "/data/logiclens/translated_safe"
Path(OUT).mkdir(exist_ok=True)

print("Загружаем модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto",
)
model.eval()
devs = Counter(str(p.device) for p in model.parameters())
for d, n in devs.items():
    print(f"  {d}: {n} тензоров")

def translate_one(text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a translator. Output ONLY the Russian translation, nothing else."},
        {"role": "user",   "content": f"Translate to Russian:\n{text}"}
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False,
    )
    if hasattr(enc, "input_ids"):
        input_ids      = enc.input_ids.to("cuda")
        attention_mask = enc.attention_mask.to("cuda")
        gen_kwargs     = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        input_ids  = enc.to("cuda")
        gen_kwargs = {"input_ids": input_ids}

    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(
            **gen_kwargs, max_new_tokens=256,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    if "</think>" in decoded:
        decoded = decoded.split("</think>", 1)[1].strip()
    return decoded or text   # fallback на оригинал если пусто

for split in ["edu_train", "edu_dev", "edu_test"]:
    out_path = f"{OUT}/{split}_ru.jsonl"
    df = pd.read_csv(f"{DATA}/{split}.csv")[["source_article", "updated_label"]]
    df["source_article"] = df["source_article"].fillna("").astype(str)
    total = len(df)

    results = []
    if Path(out_path).exists():
        results = [json.loads(l) for l in open(out_path)]
    start_idx = len(results)
    print(f"\n── {split}: {total} примеров, начинаем с {start_idx} ──")

    if start_idx >= total:
        print("  Уже готово.")
        continue

    for i in range(start_idx, total):
        orig  = df["source_article"].iloc[i]
        label = df["updated_label"].iloc[i]
        try:
            ru = translate_one(orig)
        except Exception as e:
            print(f"  ⚠ {i}: {e}")
            ru = orig
        results.append({"text": ru, "text_en": orig, "label": label})

        if i % 100 == 0:
            print(f"  {i}/{total}: {ru[:70]}")

        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  ✔ {out_path}")

print("\nГОТОВО")
