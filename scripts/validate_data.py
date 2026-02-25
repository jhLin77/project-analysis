import json
import re

RAW_PATH = "../data/raw/train.jsonl"

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
email_pattern = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

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

    # 5️⃣ PHONE / EMAIL 格式检查
    if "PHONE" in slots:
        if not phone_pattern.search(slots["PHONE"]):
            errors.append(f"{sample['conv_id']} - invalid phone format")

    if "EMAIL" in slots:
        if not email_pattern.match(slots["EMAIL"].strip()):
            errors.append(f"{sample['conv_id']} - invalid email format")

    return errors


def main():
    all_errors = []

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
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
