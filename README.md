## 📌 项目名称

**基于亚马逊商品评论的情感分析与用户关注洞察**

---

## 📖 项目背景

电商平台每天产生大量用户评论，这些文本数据中蕴含着丰富的用户体验信息。
本项目基于真实亚马逊商品评论数据，构建完整的 NLP 分析流程，对用户情感和关注点进行挖掘，帮助产品与运营团队从评论中发现关键问题并制定改进策略。

---

## 🎯 项目目标

* 自动识别用户评论的情感倾向（正面 / 负面）
* 挖掘用户差评集中关注的问题
* 通过可视化和 Dashboard 展示分析结果
* 输出可落地的产品改进建议

---

## 🛠 使用技术

* Python
* Pandas / NumPy
* Scikit-learn
* NLP（TF-IDF / 情感分析 / LDA）
* Matplotlib / Seaborn
* **Streamlit（交互式 Dashboard）**

---

## 🔍 项目流程

1. 数据获取与清洗
2. 探索性数据分析（EDA）
3. 情感分析建模
4. 用户关注点（主题）挖掘
5. 可视化分析
6. Streamlit Dashboard 展示

---

## 📊 Dashboard 展示功能

* 不同商品类别的情感分布对比
* 正 / 负面评论示例查看
* 差评高频关键词统计
* 支持交互式筛选与分析

---

## 📁 项目结构

```text
amazon_sentiment_analysis_project/
├── data/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 如何运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 项目价值

* 将非结构化文本转化为可解释的商业洞察
* 帮助产品团队快速定位用户痛点
* 展示完整的数据分析与 NLP 项目能力

---

