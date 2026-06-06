# RMDS Hackathon — Covid Vaccine Prediction 

**University of Hertfordshire | Group X**

---

## What is this project?

This project was done as part of the RMDS Hackathon. The goal was to predict whether a person got the **covid vaccine or not** (Yes = 1, No = 0) using information about their behaviour, opinions, and background.

This is a **binary classification** problem. We were given a training dataset with the answers and a test dataset without the answers. We had to build machine learning models and predict the answers for the test set.

---

## The Dataset

The dataset has **31 columns** and around **4,756 rows** for training and **4,749 rows** for testing.

| Column | What it means |
|--------|--------------|
| `respondent_id` | A unique ID for each person |
| `covid_concern` | How worried they were about covid (0–3) |
| `covid_knowledge` | How much they knew about covid (0–2) |
| `behavioral_*` | Things they did to protect themselves (e.g. wore a mask, washed hands) |
| `doctor_recc_covid` | Whether their doctor told them to get the vaccine |
| `opinion_covid_vacc_effective` | Did they think the vaccine works? (1–5) |
| `opinion_covid_risk` | Did they think they would get sick without it? (1–5) |
| `age_group`, `education`, `race`, `sex` | Background information |
| `income_poverty`, `employment_status` | Financial and work information |
| `covid_vaccine` | **Target** — did they get the vaccine? (0 = No, 1 = Yes) |

**Key things we found in the data:**
- About **67% of people did NOT get the vaccine** and only 33% did — this is called class imbalance
- `employment_sector` had nearly **49% of values missing**
- `health_insurance` had **40% of values missing**
- **Doctor recommendation** was the strongest factor — people whose doctor recommended the vaccine were much more likely to get it

---

## What We Did

### 1. Exploratory Data Analysis (EDA)
We explored the data to understand it before building any models. We made charts to look at:
- How many people got the vaccine vs did not
- Which features had missing values
- How doctor recommendation, age group, and opinions affected vaccine uptake
- Correlation between numeric features

### 2. Pre-processing
We cleaned and prepared the data for modelling:
- Added **missing indicator columns** for features with lots of missing values (so the model knows when data is missing)
- Used **median imputation** to fill in missing numbers
- Used **"Unknown"** to fill in missing categories
- Used **OneHotEncoder** to turn text categories into numbers
- Used **StandardScaler** to scale the numeric features
- Split the data into **80% training and 20% validation** (keeping the class balance the same in both)

### 3. Models We Tried
We trained 7 different models and compared them:

| Model | Notes |
|-------|-------|
| Logistic Regression | Simple and fast baseline |
| Decision Tree | Easy to understand, shows clear rules |
| Random Forest | Combines many decision trees |
| Gradient Boosting | Builds trees one after another to fix errors |
| K-Nearest Neighbors (KNN) | Predicts based on similar people |
| XGBoost | A powerful boosting model |
| LightGBM | A fast and efficient boosting model |

All models that support it were set to use `class_weight='balanced'` to handle the imbalance (more No than Yes in the data).

### 4. How We Measured Performance
We used:
- **ROC-AUC** — how well the model separates the two classes
- **F1 Score** — balances precision and recall (good for imbalanced data)
- **Accuracy** — overall correct predictions
- **5-Fold Cross Validation** — to make sure results are reliable and not just lucky

### 5. Submission
We ranked all models by F1 score and submitted the **top 5** predictions for the test set.

---

## Results

Models were ranked by F1 score. The top 5 were submitted.

---

## Files in This Repository

```
📁 Hackathon26/
│
├── dataset_C_training.csv         ← Training data (with target column)
├── dataset_C_testing.csv          ← Test data (no target column)
│
├── UH_GROUP_X.ipynb               ← Main notebook with all our code
│
```

---

## How to Run the Notebook

1. Open the notebook in **Google Colab**
2. The data is loaded directly from this GitHub repository — no need to download anything
3. Mount your Google Drive (for saving submission files)
4. Run all cells from top to bottom

The first two cells load the data straight from GitHub:

```python
datapath1 = "https://raw.githubusercontent.com/UH-GROUP-X/Hackathon26/refs/heads/main/dataset_C_training.csv"
datapath2 = "https://raw.githubusercontent.com/UH-GROUP-X/Hackathon26/refs/heads/main/dataset_C_testing.csv"
```

---

## Libraries Used

- `pandas` — loading and working with data
- `numpy` — number calculations
- `matplotlib` & `seaborn` — making charts
- `scikit-learn` — machine learning models and tools
- `xgboost` — XGBoost model
- `lightgbm` — LightGBM model
- `shap` — explainable AI (understanding why the model made each prediction)

---
