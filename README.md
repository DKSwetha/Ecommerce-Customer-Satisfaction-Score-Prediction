# DeepCSAT – Ecommerce Customer Satisfaction Score Prediction


## Project Overview

DeepCSAT is a machine learning project that predicts **Customer Satisfaction Scores (CSAT)** for e-commerce customer support interactions. Using 85,907 real support ticket records, the project builds and evaluates multiple classification models to identify dissatisfied customers and provide actionable business insights.

---

## Problem Statement

E-commerce companies handle thousands of customer support interactions daily. Predicting CSAT scores before or immediately after ticket resolution allows businesses to:
- Proactively address service gaps
- Allocate better-performing agents to high-risk ticket types
- Reduce customer churn

---

## Dataset

| Property | Detail |
|---|---|
| Records | 85,907 support tickets |
| Features | 20 columns |
| Target | CSAT Score (1–5) |
| Period | August 2023 |
| Source | eCommerce Customer Support Data |

**Key Features Used:**
- `channel_name` — Inbound, Outcall, Email
- `category` — Issue type (12 categories)
- `Tenure Bucket` — Agent experience level
- `Agent Shift` — Morning, Afternoon, Evening, Night, Split
- `Customer Remarks` — Free text feedback
- `response_time_mins` — Engineered from timestamps

---

## Key EDA Findings

- **69% of tickets score 5** — severe class imbalance
- **Email channel** has lowest mean CSAT (3.90) vs Inbound (4.25)
- **OJT agents** score lowest CSAT (4.15) vs experienced agents (4.35)
- **Cancellation tickets** are the most dissatisfying category (3.99)
- **66.5% of Customer Remarks** are missing
- Mean response time is **23 minutes**

---

## Tech Stack
```
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
nltk
shap
joblib
```

---

## Project Pipeline
```
1. Data Loading & Understanding
        ↓
2. Exploratory Data Analysis (15 charts)
        ↓
3. Data Wrangling & Feature Engineering
        ↓
4. Text Preprocessing (NLP Pipeline)
        ↓
5. Hypothesis Testing (3 tests)
        ↓
6. Preprocessing & SMOTE
        ↓
7. Model Building & Evaluation
        ↓
8. SHAP Explainability
        ↓
9. Model Saving & Deployment
```

---

## Models Built

| Model | Macro F1 | Accuracy |
|---|---|---|
| **Random Forest ** | **0.2727** | 61% |
| XGBoost | 0.2449 | 69% |
| Logistic Regression | 0.2303 | 34% |

**Random Forest** was selected as the final model based on highest Macro F1 score.

---

## NLP Pipeline

Customer Remarks were preprocessed through:
1. Contraction expansion
2. Lowercasing
3. Punctuation removal
4. URL and digit removal
5. Stopword removal
6. Lemmatization
7. TF-IDF Vectorization (top 100 features)

---

## Hypothesis Testing

| Hypothesis | Test Used | Result |
|---|---|---|
| CSAT differs across channels | One-Way ANOVA | Rejected H₀ |
| CSAT differs across tenure buckets | One-Way ANOVA | Rejected H₀ |
| Remarks presence affects CSAT | Independent t-test | Rejected H₀ |

---

## Business Insights

- Improve **Email channel** quality — lowest CSAT score
- Invest in **faster agent onboarding** — OJT agents deliver lower satisfaction
- Focus on **Cancellation and Product Query** flows — most dissatisfying categories
- Flag tickets with **customer remarks** for quality review — strong dissatisfaction signal
- Increase **Morning shift staffing** — highest volume but lowest CSAT

---

---

## Limitations

- 66.5% missing Customer Remarks reduces text signal
- Scores 2 and 3 are extremely rare — hard to predict reliably
- CSAT is inherently subjective — same interaction can get different scores
- Several potentially useful columns dropped due to 80%+ missing values

---

## Future Work

- Collect more complete data especially Customer Remarks
- Try deep learning models (BERT) for better text understanding
- Build a real-time prediction API using Flask or FastAPI
- Add more features like product category and item price
- Try binary classification (satisfied vs unsatisfied) for better performance

