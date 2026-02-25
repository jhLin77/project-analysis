import json
import re
import os

# 获取当前脚本文件所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 构造数据文件绝对路径
RAW_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "train.jsonl")


# 枚举集合
ISSUE_TYPES = {
    "无法开机","破损","不合适","尺码不符",
    "质量问题","故障","不满意","不需要",
    "延迟发货","少件"
}

REQUEST_ACTIONS = {
    "退款","退货","换货","补发",
    "改地址","开发票","人工处理","加急处理"
}

SENTIMENTS = {"生气","焦急","平静","满意"}
EVIDENCES = {"无","图片","视频","聊天记录","订单截图"}

STRONG_ENTITIES = {"ORDER_ID","PHONE","EMAIL","TRACKING_NO","DATETIME"}
SEMI_ENTITIES = {"PERSON","ADDRESS","PRODUCT","SKU_MODEL","STORE_PLATFORM"}

phone_pattern = re.compile(r"1\d{10}|1\d{2}[- ]\d{4}[- ]\d{4}")
email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

def validate_sample(sample):
    errors = []
    text = sample["full_text"]
    slots = sample["slots_ground_truth"]
    labels = sample["semantic_labels"]

    # 1️⃣ 检查 slots 是否出现在文本
    for key, value in slots.items():
        if key == "scenario":
            continue
        if value not in text:
            errors.append(f"{sample['conv_id']} - slot {key} not found in text")

    # 2️⃣ 枚举检查
    if labels["issue_type"] not in ISSUE_TYPES:
        errors.append(f"{sample['conv_id']} - invalid issue_type")

    if labels["request_action"] not in REQUEST_ACTIONS:
        errors.append(f"{sample['conv_id']} - invalid request_action")

    if labels["sentiment"] not in SENTIMENTS:
        errors.append(f"{sample['conv_id']} - invalid sentiment")

    if labels["evidence"] not in EVIDENCES:
        errors.append(f"{sample['conv_id']} - invalid evidence")

    # 3️⃣ 强实体数量
    strong_count = sum(1 for k in slots if k in STRONG_ENTITIES)
    if strong_count < 1:
        errors.append(f"{sample['conv_id']} - less than 1 strong entity")

    # 4️⃣ 半结构实体数量
    semi_count = sum(1 for k in slots if k in SEMI_ENTITIES)
    if semi_count < 2:
        errors.append(f"{sample['conv_id']} - less than 2 semi entities")

    # 5️⃣ PHONE 格式检查
    if "PHONE" in slots:
        if not phone_pattern.search(slots["PHONE"]):
            errors.append(f"{sample['conv_id']} - invalid phone format")

    # 6️⃣ EMAIL 格式检查（先提取可能的邮箱，再校验）
    if "EMAIL" in slots:
        raw_email = slots["EMAIL"].strip()
        # 尝试用 findall 提取纯邮箱字符串
        found = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_email)
        # 有提取到就取第一个，否则用原始值继续校验
        email_val = found[0] if found else raw_email

        if not email_pattern.fullmatch(email_val):
            errors.append(f"{sample['conv_id']} - invalid email format")

    return errors


def main():
    all_errors = []

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            # 先去掉前后空白字符
            line = line.strip()
            # 如果这一行是空字符串（空行），就跳过
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"🚫 第 {i} 行 JSON 解析错误:", e)
                continue

            errs = validate_sample(sample)
            all_errors.extend(errs)

    if all_errors:
        print("发现错误：")
        for e in all_errors:
            print(e)
    else:
        print("✅ 所有数据校验通过！")


if __name__ == "__main__":
    main()
