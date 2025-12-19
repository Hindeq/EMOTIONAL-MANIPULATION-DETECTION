# Emotional Manipulation Detection in Tweets

This project aims to detect **emotional manipulation in short texts (tweets)** using **BERTweet** and a **multi-task learning** approach. The model predicts both the **emotion** and **manipulation/bias** of a text in a single pass.

## Methodology

- **Datasets:** Three datasets were used for emotion, manipulation, and bias classification. Preprocessing included cleaning text, standardizing labels, balancing classes, and splitting into train/validation/test sets.
- **Model:** A multi-task model based on **vinai/bertweet-base** was used with two classification heads:
  - **Emotion Head:** 6 classes – Sadness, Joy, Love, Anger, Fear, Surprise
  - **Manipulation/Bias Head:** Binary classification
- **Training:** The model was trained using a combined loss, Adam optimizer, cross-entropy loss, and class weights to handle imbalance.

## Results

- **Emotion classification:** Accuracy ≈ 96%, high macro-F1 score  
- **Manipulation/Bias classification:** Accuracy ≈ 86%, successfully detects most manipulative or biased cases  
- Confusion matrices highlight areas where the model struggles with ambiguous or overlapping classes.
