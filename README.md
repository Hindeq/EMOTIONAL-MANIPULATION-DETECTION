# 🐦 Emotional Manipulation Detection in Tweets
**Multi-task BERTweet Model with Streamlit Interface**

---

## 📌 Project Overview
This project presents an **end-to-end NLP system** for detecting **emotional manipulation in short texts (tweets)**.  
A **multi-task deep learning model** based on **BERTweet** jointly predicts:

- **Dominant emotion** (6 classes)
- **Manipulation / propaganda / bias** (binary)

The system is deployed through an **interactive Streamlit application** and supports full evaluation on test datasets.

---

## 📊 Datasets & Preprocessing
Three datasets were used:

| Dataset | Task Type | Classes |
|------|----------|--------|
| Emotion | Multiclass | Sadness, Joy, Love, Anger, Fear, Surprise |
| Propaganda | Binary | Manipulative / Non-manipulative |
| Bias | Binary | Biased / Non-biased |

### Preprocessing Steps
- Text cleaning (URLs, emojis, mentions, special characters)
- Label standardization
- Deduplication
- Partial class balancing
- Train / validation / test split

Cleaned datasets:
data_clean/

├── emotion_clean.csv

├── propaganda_clean.csv

├── bias_clean.csv


---

## 🧠 Model Architecture
- **Backbone**: `vinai/bertweet-base`
- Pretrained on tweets and short social media texts
- Two task-specific classification heads:
  - **Emotion Head** → 6 classes
  - **Manipulation/Bias Head** → Binary

✔️ Both predictions are produced in a **single forward pass**.

---

## ⚙️ Training Details
- Optimizer: Adam
- Learning rate: `2e-5`
- Batch size: `32`
- Epochs: ~10–15
- Loss function:
loss_total = loss_emotion + loss_manipulation

- Cross-entropy loss with class weighting for emotion imbalance

---

## 📈 Results
| Task | Accuracy | Notes |
|----|--------|------|
| Emotion | ~96% | Strong performance across all classes |
| Manipulation / Bias | ~86% | Effective detection of manipulative patterns |

Misclassifications mainly occur in ambiguous or overlapping cases.

---

## 🖥 Streamlit Application
The Streamlit app provides:

- 📊 Dataset exploration & visualization
- 🔮 Real-time text prediction
- 📏 Model evaluation on test datasets
- ℹ️ Model & training overview

Run locally:
streamlit run app.py

---

## 🐳 Docker Support
The project includes Docker support for reproducibility.
docker build -t emotional-manipulation-app .
docker run -p 8501:8501 emotional-manipulation-app


---

## 🎯 Key Contributions
- Multi-task NLP model for emotion & manipulation detection
- BERTweet-based architecture optimized for short texts
- Hybrid decision logic improving interpretability
- Full pipeline from data to deployment

> *This project demonstrates how deep learning and linguistic reasoning can be combined to analyze emotional manipulation in social media content.*

