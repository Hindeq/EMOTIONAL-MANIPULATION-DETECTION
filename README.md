# 🐦 Emotional Manipulation Detection in Tweets

This project aims to detect **emotional manipulation in short texts (tweets)** using **BERTweet** and a **multi-task learning** approach. The model predicts both the **emotion** and **manipulation/bias** of a text in a single forward pass, making it suitable for real-world social media analysis.

---

## 🧪 Methodology

### 📊 Datasets
Three datasets were used to capture different aspects of short-text analysis:

- **Emotion Dataset 😢😊❤️😡😱😮** – Multiclass (6 emotions)  
- **Manipulation/Propaganda Dataset ⚖️** – Binary classification (manipulative vs non-manipulative)  
- **Bias Dataset ⚖️** – Binary classification (biased vs non-biased)  

**Preprocessing steps included:**  
1. **Text Cleaning:** Removal of URLs, mentions, emojis, and special characters  
2. **Label Standardization:** Consistent representation of class labels across datasets  
3. **Deduplication:** Removing repeated entries to avoid bias  
4. **Class Balancing:** Mitigating data skew for underrepresented classes  
5. **Data Splitting:** Dividing data into training, validation, and test sets  

Cleaned datasets were stored as `emotion_clean.csv`, `propaganda_clean.csv`, and `bias_clean.csv`.

---

### 🧠 Model Architecture

A **multi-task learning model** was built using **vinai/bertweet-base** as the backbone, with two classification heads:

- **Emotion Head :** 6 classes – Sadness, Joy, Love, Anger, Fear, Surprise  
- **Manipulation/Bias Head :** Binary classification – 0: non-manipulative/non-biased, 1: manipulative/biased  

**Key advantages of this architecture:**  
- Simultaneous prediction of emotion and manipulation in one forward pass  
- Leverages shared representations to improve generalization  
- Captures subtle emotional cues and manipulative patterns in tweets  

---

### ⚙️ Training Details

- **Loss Function:** Combined loss – `loss_total = loss_emotion + loss_manipulation`  
- **Optimizer:** Adam  
- **Loss Type:** Cross-entropy with softmax for both heads  
- **Class Weights:** Applied to handle imbalanced emotion classes  
- **Epochs:** Trained until convergence (~10–15 epochs)  
- **Batch Size:** 32  
- **Learning Rate:** 2e-5  

---

## 📈 Results

| Task | Accuracy | Macro-F1 | Notes |
|------|----------|----------|-------|
| Emotion  | 96% | High | Strong performance across all emotion classes |
| Manipulation/Bias  | 86% | Moderate/High | Successfully detects most manipulative or biased cases |

**Insights:**  
- Confusion matrices show that misclassifications mostly occur in **ambiguous or overlapping cases**.  
- The multi-task framework effectively captures both emotional and manipulative cues.  

---

## 🚀 Deployment

The model is deployed via **Streamlit** for interactive testing and **Docker** for reproducibility:

- **Streamlit:** Run locally with `streamlit run app.py`  
- **Docker:** Build and run the container using the provided `Dockerfile`  
