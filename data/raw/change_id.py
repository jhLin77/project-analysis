import json

input_file = "train_raw_L3_list.jsonl"      # 你的原始文件
output_file = "train_raw_L3.jsonl"    # 修改后的文件

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    i = 1

    for line in f_in:
        line = line.strip()

        # ✅ 跳过空行
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print("跳过非法行:", line[:50])
            continue

        data["conv_id"] = f"conv_L3_{i}"
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

        i += 1

print("修改完成！")