# 📰 113_2-DLA_news_politics_classifiers

This project investigates the **automatic classification of political ideology** in Taiwanese news articles. We focus on identifying three stances:

 **Pro-Green** | **Pro-Blue** | **Neutral**

The goal is to detect framing bias in media by using **deep learning techniques**, particularly **soft-label knowledge distillation**, to improve ideological classification in Mandarin-language political journalism.

---

## 📚 Dataset

We collected a total of **3,185 articles** from the following **Taiwanese mainstream media outlets** during **April–May 2025**:

- TVBS 新聞  
- 民視新聞 (FTV)  
- 三立新聞 (SET)  
- 公視新聞 (PTS)  

### 🏷️ Labeling Process

Each article was labeled as **pro-green**, **neutral**, or **pro-blue** using a **three-model voting scheme**:

1. **GPT-4.1-mini** (OpenAI, 2025)  
2. **Claude 3.5 Haiku** (Anthropic, 2024)  
3. **Gemini 2.5 Flash** (DeepMind, 2025)  

Each model acted as a **Taiwan-political-news expert** and voted based on title and content. Final labels were assigned via **majority vote**. Around **2.5% (81 articles)** with disagreement were **manually reviewed** by human annotators.

### 🧹 Preprocessing

- Duplicate removal  
- Noise filtering (e.g., ads, repeated media names)  
- **Translation**: All articles were translated from **Traditional Chinese to English** using **GPT-4.1-mini**, preserving political tone and journalistic phrasing.

➡️ Final dataset: **3,166 unique labeled articles**

---

## 🎯 Task Definition

A **3-class classification** problem:

> Given a news article (title + content), predict its political stance.

**Classes:** `{pro-green, pro-blue, neutral}`

---

## 🧠 Proposed Method: Soft-Label Knowledge Distillation

We use the **English-language POLITICS model** (Liu et al., 2022) as a **fixed teacher** and **fine-tune a CKIP BERT model** (trained on Traditional Chinese) as the **student**.

### 🏗️ Architecture Overview

- **Teacher Input**: Translated English news  
- **Student Input**: Original Traditional Chinese news  
- **Objective**: Match the **teacher’s output probability distribution** (soft labels)

### 🧮 Loss Function

The total loss combines:

- **KL-Divergence Loss** from teacher soft labels  
- **Cross-Entropy Loss** from ground-truth labels

```python
L_total = α * L_KD + (1 - α) * L_CE
```python

## 🧪 Experiments

- **Cross-validation**: 5-fold  
- **Frameworks**: PyTorch + HuggingFace  
- **GPU**: Kaggle P100  
- **Optimizer**: AdamW (learning rate = 2e-5)  
- **Batch size**: 16  
- **Epochs**: 5  
- **Max sequence length**: 512  

### 📊 Performance (Selected)

| Model           | Macro Acc | F1    | Precision | Recall |
|-----------------|-----------|-------|-----------|--------|
| CKIP BERT       | 0.767     | 0.755 | 0.755     | 0.756  |
| KD-CKIP (Soft)  | 0.751     | 0.743 | 0.738     | 0.754  |

> ⚠️ Although soft-label knowledge distillation yielded competitive results, it did not surpass direct fine-tuning.

---

## 📌 Key Insight

Soft-label distillation allows transfer of nuanced ideological bias detection from **English political models** to **Chinese-language models**, but **semantic drift from translation** may still limit its effectiveness.
