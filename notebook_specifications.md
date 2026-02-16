
# Ensemble Model for Robust Corporate Bond Default Prediction

### Introduction

You are a Credit Analyst at a leading asset management firm tasked with enhancing the robustness of corporate bond default predictions. Your organization, recognizing the limitations of individual models, aims to integrate diverse data sources and modeling approaches for a more stable and compliant risk assessment solution.

## Notebook Specification

### 1. Install Required Libraries

```python
!pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

### 2. Import Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
```

### 3. Data Preparation

#### a. Markdown Cell — Story + Context + Real-World Relevance

As a Credit Analyst, the first step is preparing your dataset. You’ll need to ensure it includes all relevant features, even if some need to be simulated. This aligns with real-world requirements of handling data gaps in financial datasets.

#### b. Code Cell

```python
# Load dataset
df = pd.read_csv('credit_data.csv')

# Define feature subsets
fundamental_features = ['fico_score', 'dti', 'income', 'loan_amount', 'ltv', 'delinquencies_2yr', 'open_accounts', 'revolving_utilization', 'employment_length', 'home_ownership_encoded']
market_features = ['equity_volatility_60d', 'stock_momentum_12m', 'credit_spread_sector', 'interest_rate_sensitivity', 'market_cap_quintile']
nlp_features = ['earnings_sentiment_score', 'tone_shift_qoq', 'risk_topic_regulatory', 'risk_topic_operational', 'risk_topic_financial', 'filing_text_complexity']

# Simulate missing features
np.random.seed(42)
for col in market_features + nlp_features:
    if col not in df.columns:
        df[col] = np.random.randn(len(df)) * 0.5 + df['default'].values * np.random.uniform(0.1, 0.3)

# Split data
X = df[fundamental_features + market_features + nlp_features]
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

### 4. Train Diverse Base Models

#### a. Markdown Cell — Story + Context + Real-World Relevance

You will train three distinct models to capture different dimensions of default risk. This step mimics real-world workflows where analysts explore various data aspects to gain holistic insights.

#### b. Code Cell

```python
# Logistic Regression on Fundamental Features
scaler_fund = StandardScaler()
X_train_fund = scaler_fund.fit_transform(X_train[fundamental_features])
X_test_fund = scaler_fund.transform(X_test[fundamental_features])
model_fund = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
model_fund.fit(X_train_fund, y_train)
prob_fund = model_fund.predict_proba(X_test_fund)[:, 1]
auc_fund = roc_auc_score(y_test, prob_fund)

# XGBoost on Market Features
model_mkt = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric='auc', random_state=42)
model_mkt.fit(X_train[market_features], y_train)
prob_mkt = model_mkt.predict_proba(X_test[market_features])[:, 1]
auc_mkt = roc_auc_score(y_test, prob_mkt)

# LightGBM on NLP Features
model_nlp = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, is_unbalance=True, random_state=42)
model_nlp.fit(X_train[nlp_features], y_train)
prob_nlp = model_nlp.predict_proba(X_test[nlp_features])[:, 1]
auc_nlp = roc_auc_score(y_test, prob_nlp)
```

### 5. Implement Ensemble Strategies

#### a. Markdown Cell — Story + Context + Real-World Relevance

Combining predictions from multiple models through ensembling improves robustness, a critical requirement in your job to meet regulatory compliance like SR 11-7.

#### b. Code Cell

```python
# Simple Probability Averaging
prob_avg = (prob_fund + prob_mkt + prob_nlp) / 3
auc_avg = roc_auc_score(y_test, prob_avg)

# Voting Classifier
voting_soft = VotingClassifier(estimators=[('fund', model_fund), ('mkt', model_mkt), ('nlp', model_nlp)], voting='soft')
voting_soft.fit(X_train, y_train)
prob_vote = voting_soft.predict_proba(X_test)[:, 1]
auc_vote = roc_auc_score(y_test, prob_vote)

# Stacking with Meta-Learner
stacking = StackingClassifier(estimators=[('fund', model_fund), ('mkt', model_mkt), ('nlp', model_nlp)], final_estimator=LogisticRegression(), cv=5)
stacking.fit(X_train, y_train)
prob_stack = stacking.predict_proba(X_test)[:, 1]
auc_stack = roc_auc_score(y_test, prob_stack)
```

### 6. Evaluate Performance & Stability

#### a. Markdown Cell — Story + Context + Real-World Relevance

Evaluating models' performance ensures they are both accurate and reliable across various economic periods, a priority for effective risk management.

#### b. Code Cell

```python
# Print AUC scores
print("AUC scores:")
print(f"Fundamental Model: {auc_fund:.4f}")
print(f"Market Signal Model: {auc_mkt:.4f}")
print(f"NLP Model: {auc_nlp:.4f}")
print(f"Average Ensemble: {auc_avg:.4f}")
print(f"Voting Ensemble: {auc_vote:.4f}")
print(f"Stacking Ensemble: {auc_stack:.4f}")

# Bootstrap for stability
def bootstrap_auc(y_true, y_prob, n_boot=1000):
    aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        aucs.append(roc_auc_score(y_true.iloc[idx], y_prob[idx]))
    return np.percentile(aucs, [2.5, 97.5])

ci_stack = bootstrap_auc(y_test, prob_stack)
print(f"Stacking Ensemble 95% CI: [{ci_stack[0]:.4f}, {ci_stack[1]:.4f}]")
```

### 7. Analyze Model Agreement & Disagreement

#### a. Markdown Cell — Story + Context + Real-World Relevance

This analysis helps prioritize high-value investigations by identifying cases requiring human review when model predictions disagree.

#### b. Code Cell

```python
# Agreement analysis
threshold = 0.15
pred_fund = (prob_fund > threshold).astype(int)
pred_mkt = (prob_mkt > threshold).astype(int)
pred_nlp = (prob_nlp > threshold).astype(int)

# Agreement score
agreement = pred_fund + pred_mkt + pred_nlp

# Default rate by agreement
confidence_df = pd.DataFrame({'agreement': agreement, 'true_default': y_test.values, 'prob_stack': prob_stack})
print("Default Rate by Model Agreement:")
for n_agree in [0, 1, 2, 3]:
    mask = confidence_df['agreement'] == n_agree
    if mask.sum() > 0:
        default_rate = confidence_df.loc[mask, 'true_default'].mean()
        print(f"{n_agree}/3 models flag default: {mask.sum()} cases, actual default rate = {default_rate:.2%}")
```

