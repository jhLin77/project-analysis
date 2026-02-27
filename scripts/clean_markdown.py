import json
import re
import os
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "train_raw.jsonl")
TARGET_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "train.jsonl")

extract_email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}")

def clean_email_field(email_raw: str) -> str:
    found = extract_email_pattern.findall(email_raw)
    return found[0] if found else email_raw.strip()

def main():
    changed_count = 0
    processed_count = 0

    target_dir = os.path.dirname(TARGET_PATH)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target_dir, delete=False, prefix="tmp_train_", suffix=".jsonl"
    ) as tmp_file:
        tmp_path = tmp_file.name

        with open(RAW_PATH, "r", encoding="utf-8") as fr:
            for line_num, raw_line in enumerate(fr, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 解析第 {line_num} 行失败 (跳过): {e}")
                    continue

                slots = sample.get("slots_ground_truth", {})
                if "EMAIL" in slots:
                    original_email = slots["EMAIL"]
                    cleaned_email = clean_email_field(original_email)
                    if cleaned_email != original_email:
                        print(f"🔁 修改第 {line_num} 行的 EMAIL: {original_email} → {cleaned_email}")
                        slots["EMAIL"] = cleaned_email
                        changed_count += 1

                tmp_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"📊 进度: {processed_count} 条数据已处理…")

    os.replace(tmp_path, TARGET_PATH)

    print(f"✅ 清理结束，总共处理 {processed_count} 条数据")
    print(f"📦 修改了 {changed_count} 条 EMAIL")
    print(f"📄 新 train.jsonl 已覆盖旧文件: {TARGET_PATH}")

if __name__ == "__main__":
    main()