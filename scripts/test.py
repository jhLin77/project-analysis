import json

file_path = "data/raw/train.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        # 跳过完全空白的行
        if not line:
            print(f"⚠ 空行在第 {i} 行被跳过")
            continue

        try:
            # 尝试把这一行解析成 JSON
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"❌ 第 {i} 行 JSON 解析错误: {e}")
        else:
            # 如果没有错误也可以输出
            print(f"第 {i} 行 格式正确")