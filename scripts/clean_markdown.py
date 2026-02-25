import json
import re
import os

# 数据文件路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "train.jsonl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "train_clean.jsonl")

# 提取 email 的正则
extract_email_pattern = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

def clean_email_field(email_raw: str) -> str:
    """
    从可能包含 Markdown / 噪音的字符串中提取纯邮箱，
    如果没有提取到，则返回原始值。
    """
    found = extract_email_pattern.findall(email_raw)
    return found[0] if found else email_raw.strip()

def main():
    changed_count = 0

    with open(RAW_PATH, "r", encoding="utf-8") as fr, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fw:

        for line in fr:
            if not line.strip():
                continue  # 跳过空行

            sample = json.loads(line)
            slots = sample.get("slots_ground_truth", {})

            # 只处理有 EMAIL 字段的样本
            if "EMAIL" in slots:
                raw_val = slots["EMAIL"]
                cleaned = clean_email_field(raw_val)

                if cleaned != raw_val:
                    # 打印变化对比提示
                    print(f"🔁 {sample['conv_id']} EMAIL 改写:")
                    print(f"    处理前: {repr(raw_val)}")
                    print(f"    处理后: {repr(cleaned)}\n")

                    slots["EMAIL"] = cleaned
                    changed_count += 1

            # 写入新的 train_clean.jsonl
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 清理完成，总共改写 {changed_count} 条 EMAIL 字段。")
    print(f"📦 清理后的文件已保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()