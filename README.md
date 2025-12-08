# README – Amazon Fine Food Reviews Sentiment Analysis

## 📌 Overview
This project applies Natural Language Processing (NLP) and Machine Learning to classify Amazon Fine Food Reviews as **positive** or **negative**.  
The workflow follows the **CRISP-DM framework**, moving from business understanding to deployment.  
The final model is deployed as a **Streamlit web application**, allowing users to input a review and receive a sentiment prediction.

## 📂 Project Structure
- `amazon_review_sentiment_eda.ipynb` — Exploratory Data Analysis  
- `amazon_review_sentiment_modeling_BALANCED.ipynb` — Modeling on balanced dataset  
- `amazon_review_sentiment_modeling_UNBALANCED.ipynb` — Modeling on unbalanced dataset  
- `app.py` — Streamlit sentiment classifier  
- `logreg_tfidf_cv_best.joblib` — Final deployed model  

# 🧠 CRISP-DM PROCESS

## 1. Business Understanding
The aim is to build a model capable of automatically classifying sentiment in food reviews.  
Use cases include customer feedback monitoring, product evaluation, and automated scoring.

## 2. Data Understanding
- Dataset: Amazon Fine Food Reviews  
- Includes text reviews and rating scores  
- Initial analysis revealed class imbalance and textual noise  
- EDA involved word clouds, n-gram frequency analysis, and distribution inspection

## 3. Data Preparation
- Text cleaning: lowercasing, punctuation removal, tokenization, stopword removal  
- Feature engineering:
  - **TF-IDF** vectorisation (1–2 grams)
  - Additional numeric features (word_count, text_length)  
- Two datasets created:
  - **Unbalanced** (original distribution)
  - **Balanced** (equal positive and negative samples)

## 4. Modeling
The following machine-learning models were trained and tuned:
- **Logistic Regression (TF-IDF)**
- **Linear SVM (TF-IDF)**
- **ColumnTransformer + Logistic Regression**
- **MLP (TF-IDF → SVD → Neural Network)**  

All models were evaluated using **5-fold Stratified Cross-Validation**, reporting:
- Macro F1-score  
- Precision, Recall  
- Confusion Matrix  
- ROC-AUC & PR-AUC  

**Best model:**  
✔ **TF-IDF + Logistic Regression**  
Selected for deployment due to highest macro-F1 and stability.

## 5. Evaluation
Evaluation compared:
- Balanced vs. Unbalanced dataset performance  
- Out-of-fold predictions (no leakage)
- ROC-AUC and PR-AUC behaviour  
- Error patterns using confusion matrices  

## 6. Deployment
A Streamlit web app provides:
- Sentiment prediction (positive/negative)
- Model confidence scores  

The deployed model is:  
**`logreg_tfidf_cv_best.joblib`**

## ✔ Final Notes
This repository demonstrates a full CRISP-DM workflow for sentiment analysis, covering:
- Exploratory analysis  
- Preprocessing  
- Feature engineering  
- Model training  
- Hyperparameter tuning  
- System evaluation  
- Web deployment  

Suitable for both academic and production-level applications.


## 6. Repo structure 

```text
nlp_sentiment_analysis/
├── notebooks/
│   ├── amazon_review_sentiment_eda.ipynb
│   ├── amazon_review_sentiment_modeling_BALANCED.ipynb
│   └── amazon_review_sentiment_modeling_UNBALANCED.ipynb
│
├── models/
│   └── logreg_tfidf_cv_best.joblib        # final deployed model
│
├── app/
│   └── app.py                             # Streamlit sentiment app
│
├── data/                           
│   └── download directly from kaggle    
│
├── README.md
└── requirements.txt
```

---

## 7. Requirements

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
wordcloud
joblib
streamlit
```

---

**Note:** The full Kaggle dataset is large; download it directly from Kaggle inside Colab/Kaggle and point the notebook to `/kaggle/input/amazon-fine-food-reviews/Reviews.csv`.
