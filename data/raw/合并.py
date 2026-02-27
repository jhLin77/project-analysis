import os

# 你的三个文件
input_files = [
    r"train_raw_L1_with_group.jsonl",
    r"train_raw_L2_with_group.jsonl",
    r"train_raw_L3_with_group.jsonl"
]

# 输出文件
output_file = r"merged.jsonl"

with open(output_file, "w", encoding="utf-8") as outfile:
    for file in input_files:
        print(f"正在处理: {file}")

        if not os.path.exists(file):
            print(f"文件不存在: {file}")
            continue

        with open(file, "r", encoding="utf-8-sig") as infile:
            for line in infile:
                line = line.strip()
                if line:   # 跳过空行
                    outfile.write(line + "\n")

print("合并完成！")