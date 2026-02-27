import json

files = {
    r"train_raw_L1.jsonl": "L1",
    r"train_raw_L2.jsonl": "L2",
    r"train_raw_L3.jsonl": "L3"
}

for input_file, group_value in files.items():

    output_file = input_file.replace(".jsonl", "_with_group.jsonl")

    with open(input_file, "r", encoding="utf-8-sig") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            data["group"] = group_value
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"{input_file} 处理完成")

print("全部文件处理完成！")