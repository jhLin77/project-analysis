import json
import os

RAW_PATH = "../data/raw/train.jsonl"
OUTPUT_PATH = "../data/labeled/train.jsonl"

# 所有实体类型
ENTITY_TYPES = {
    "ORDER_ID",
    "PHONE",
    "EMAIL",
    "TRACKING_NO",
    "DATETIME",
    "PERSON",
    "ADDRESS",
    "PRODUCT",
    "SKU_MODEL",
    "STORE_PLATFORM"
}

def find_all_spans(text, value):
    """
    找到文本中所有 value 出现的位置
    """
    spans = []
    start = 0
    while True:
        idx = text.find(value, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(value)))
        start = idx + len(value)
    return spans


def generate_entities(sample):
    text = sample["full_text"]
    slots = sample["slots_ground_truth"]
    entities = []

    for key, value in slots.items():
        if key not in ENTITY_TYPES:
            continue

        spans = find_all_spans(text, value)

        if not spans:
            print(f"⚠ 未找到 span: {sample['conv_id']} - {key}")
            continue

        for start, end in spans:
            entities.append({
                "type": key,
                "start": start,
                "end": end,
                "text": value
            })

    return entities


def main():
    os.makedirs("../data/labeled", exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            sample = json.loads(line)
            entities = generate_entities(sample)
            sample["entities"] = entities
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("✅ Span 生成完成，已保存到 labeled/train.jsonl")


if __name__ == "__main__":
    main()
