import json
import numpy as np
import os
from collections import Counter

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from seqeval.metrics import precision_score, recall_score, f1_score


# ====== 路径设置 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/labeled/train.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "../ner_model")


# ====== 读取数据 ======
texts = []
entities_list = []
groups_list = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)
        texts.append(sample["full_text"])
        entities_list.append(sample["entities"])
        groups_list.append(sample["group"])   # ⭐ 新增


# ====== 构建标签集合 ======
label_set = set()
for entities in entities_list:
    for ent in entities:
        label_set.add(ent["type"])

label_list = ["O"]
for label in sorted(label_set):
    label_list.append("B-" + label)
    label_list.append("I-" + label)

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

print("标签集合:", label_list)


# ====== tokenizer ======
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# ====== 对齐函数 ======
def tokenize_and_align(example):
    encoding = tokenizer(
        example["text"],
        return_offsets_mapping=True,
        truncation=True,
        padding=False,
        max_length=256
    )

    offsets = encoding["offset_mapping"]
    labels = ["O"] * len(offsets)

    for i, (s, e) in enumerate(offsets):
        if s == e:
            labels[i] = -100

    for ent in example["entities"]:
        start, end, ent_type = ent["start"], ent["end"], ent["type"]

        overlapped = []
        for i, (s, e) in enumerate(offsets):
            if labels[i] == -100:
                continue
            if not (e <= start or s >= end):
                overlapped.append(i)

        if not overlapped:
            continue

        labels[overlapped[0]] = "B-" + ent_type
        for i in overlapped[1:]:
            labels[i] = "I-" + ent_type

    encoding["labels"] = [label2id[x] if x != -100 else -100 for x in labels]
    encoding.pop("offset_mapping")
    return encoding


# ====== 构建 Dataset ======
dataset = Dataset.from_dict({
    "text": texts,
    "entities": entities_list,
    "group": groups_list   # ⭐ 新增
})

dataset = dataset.map(tokenize_and_align, batched=False)


# ====== Debug 标签分布 ======
def debug_label_stats(ds, n=5):
    cnt = Counter()
    for i in range(min(n, len(ds))):
        lab = ds[i]["labels"]
        for x in lab:
            if x == -100:
                continue
            cnt[id2label[x]] += 1
    print("【DEBUG】前几条样本 label 分布（不含-100）Top:", cnt.most_common(10))

debug_label_stats(dataset, n=10)


# ==================================================
# ⭐⭐ 关键修改：OOD 切分 ⭐⭐
# ==================================================

train_dataset = dataset.filter(lambda x: x["group"] in ["L1", "L2"])
eval_dataset = dataset.filter(lambda x: x["group"] == "L3")

print("Train size:", len(train_dataset))
print("Eval size:", len(eval_dataset))


# ====== 加载模型 ======
model = BertForTokenClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)


# ====== 评估函数 ======
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, lab in zip(predictions, labels):
        cur_labels = []
        cur_preds = []
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                cur_labels.append(id2label[l_i])
                cur_preds.append(id2label[p_i])
        true_labels.append(cur_labels)
        true_predictions.append(cur_preds)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


# ====== 训练参数 ======
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=10,
)


# ====== Trainer ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

print("训练完成！模型保存在:", OUTPUT_DIR)