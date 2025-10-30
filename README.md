# nlp_sentiment_analysis

Sentiment analysis project on the **Amazon Fine Food Reviews** dataset. The goal (so far) is to clean the raw Kaggle data, engineer basic text features, explore the class/length distributions, and get the data ready for modelling (TF–IDF + classifiers).

---

## 1. Dataset
- Source: Kaggle – *Amazon Fine Food Reviews* (`Reviews.csv`)
- Original shape: **568,454 rows × 10 columns**
- Key fields: `Score` (1–5), `Text`, `Summary`, `Time`, `ProductId`, `UserId`

---

## 2. Cleaning steps

We applied the following in the notebook:

1. **Drop empty reviews**
   ```python
   df = df.dropna(subset=["Text"])
   ```
   Ensures every row has actual text.

2. **Drop duplicate reviews**
   ```python
   df = df.drop_duplicates(subset=["Text"])
   ```
   Avoids training/evaluating on the exact same review text.

3. **Minimum length filter (text quality)**
   - First tried `< 10` words, then used **`word_count >= 20`**
   - Rationale: very short reviews were mostly generic (“Great”, “Loved it”), low information, and many were similar.

4. **Remove extreme-length outliers**
   ```python
   df = df[df["word_count"] <= 1000]
   ```
   Only **133** rows were dropped (≈0.03%), spread across classes → negligible effect but tidier distribution.

5. **Create extra features**
   ```python
   df["text_len"] = df["Text"].str.len()
   df["word_count"] = df["Text"].str.split().str.len()
   df["review_time"] = pd.to_datetime(df["Time"], unit="s")
   ```
   These are used in EDA and later can go into a `ColumnTransformer`.

6. **Create 4-class sentiment from `Score`**
   ```python
   def score_to_sentiment_4(s):
       if s <= 2: return "negative"
       elif s == 3: return "neutral"
       elif s == 4: return "positive"
       else: return "very_positive"

   df["sentiment_4"] = df["Score"].apply(score_to_sentiment_4)
   ```

**Resulting shape:** **382,120 rows × 13 columns**  
(so we removed **186,334** low-quality / duplicate / outlier rows)

---

## 3. EDA highlights

- **Length**: median ≈ **58 words** (IQR ≈ 35–99) → most reviews are short–medium.
- **Scores**: heavily **skewed to 4–5 stars** (median = 5) → dataset is **imbalanced**.
- **Helpfulness**: mostly 0–2, very sparse.
- **Time**: reviews from **1999 → 2012**, concentrated around **2010–2012**.

Because of the score skew, later modelling should report **macro-F1** and/or use **`class_weight="balanced"`**.

---

## 4. Plots (EDA)

We generated simple plots to confirm the exploration:

1. **Rating / Score distribution**
   ```python
   df["Score"].value_counts().sort_index().plot(kind="bar")
   ```
   Shows the 4–5 star dominance.

2. **Word count histogram**
   ```python
   df["word_count"].plot(kind="hist", bins=50)
   ```
   Shows most reviews under 100 words and a long tail (which we clipped).

3. **Reviews per year (optional)**
   ```python
   df.groupby(df["review_time"].dt.year)["Id"].count().plot(kind="bar")
   ```

These plots support the cleaning decisions (drop short, drop extreme long, handle imbalance).

---

## 5. Next steps (not in this repo yet)
- Train/test split (stratified by `sentiment_4`)
- TF–IDF vectorisation of `Text`
- Baseline model: Logistic Regression
- Stronger model: Linear SVM with `class_weight="balanced"`
- Compare to auto/benchmark (e.g. AI Studio) and report macro-F1

---

## 6. Repo structure (suggested)

```text
nlp_sentiment_analysis/
├── notebooks/
│   └── amazon_review_sentiment_eda.ipynb
├── data/           # optional, small sample only
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
```

---

**Note:** The full Kaggle dataset is large; download it directly from Kaggle inside Colab/Kaggle and point the notebook to `/kaggle/input/amazon-fine-food-reviews/Reviews.csv`.
