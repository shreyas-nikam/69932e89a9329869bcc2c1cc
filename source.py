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
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# --- Helper Class for Feature Subset Selection ---
class FeatureSubsetClassifier(ClassifierMixin, BaseEstimator):
    """
    A wrapper to select specific features before passing them to an estimator.
    Useful for ensembles like VotingClassifier or StackingClassifier where
    base estimators operate on different feature subsets.
    """
    _estimator_type = "classifier"

    def __init__(self, estimator, features):
        self.estimator = estimator
        self.features = features
        self.estimator_ = None  # To store the fitted estimator

    def fit(self, X, y):
        # Clone the estimator to ensure it's a fresh instance for fitting
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[self.features], y)
        # Required by sklearn meta-estimators for consistency
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        else:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X[self.features])

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X[self.features])

    def get_params(self, deep=True):
        """Get parameters for this estimator and wrapped estimator."""
        params = super().get_params(deep=True)
        if deep:
            estimator_params = self.estimator.get_params(deep=True)
            for k, v in estimator_params.items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator and wrapped estimator."""
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        # Separate params for FeatureSubsetClassifier and its estimator
        local_params = {}
        estimator_params = {}

        for key, value in params.items():
            if key in valid_params: # Check if it's a direct parameter of FeatureSubsetClassifier
                local_params[key] = value
            elif key.startswith('estimator__') and key[len('estimator__'):] in self.estimator.get_params(deep=True):
                estimator_params[key[len('estimator__'):]] = value
            else:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self.__class__.__name__!r}. "
                    f"Check the list of available parameters with `estimator.get_params().keys()`."
                )
        
        # Set parameters for FeatureSubsetClassifier
        if "estimator" in local_params:
            self.estimator = local_params.pop("estimator")
        if "features" in local_params:
            self.features = local_params.pop("features")
        
        # Call super set_params for any remaining direct parameters (e.g., if we had any other than estimator/features)
        # In this simple case, we don't have other direct parameters, but good practice.
        # This part could be simplified if only `estimator` and `features` are direct params.

        # Set parameters for the wrapped estimator
        if estimator_params:
            self.estimator.set_params(**estimator_params)
            
        return self

# --- Data Simulation Function ---
def simulate_credit_data(n_samples=10000, random_state=42):
    """
    Simulates a credit dataset with fundamental, market, and NLP features.

    Args:
        n_samples (int): Number of samples to generate.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The simulated credit data.
            - list: List of fundamental feature names.
            - list: List of market feature names.
            - list: List of NLP feature names.
    """
    np.random.seed(random_state)
    rng = np.random.default_rng(random_state)

    # 10% default rate
    default = np.random.binomial(1, 0.10, n_samples)

    data = {
        'fico_score': np.random.randint(300, 850, n_samples),
        'dti': np.random.uniform(0.1, 0.5, n_samples),
        'income': np.random.uniform(30000, 150000, n_samples),
        'loan_amount': np.random.uniform(5000, 50000, n_samples),
        'ltv': np.random.uniform(0.1, 0.9, n_samples),
        'delinquencies_2yr': np.random.randint(0, 5, n_samples),
        'open_accounts': np.random.randint(2, 20, n_samples),
        'revolving_utilization': np.random.uniform(0.01, 0.99, n_samples),
        'employment_length': np.random.randint(1, 30, n_samples),
        'home_ownership_encoded': np.random.randint(0, 3, n_samples),
        'default': default
    }
    df = pd.DataFrame(data)

    fundamental_features = ['fico_score','dti','income','loan_amount','ltv','delinquencies_2yr',
                            'open_accounts','revolving_utilization','employment_length','home_ownership_encoded']

    market_features = ['equity_volatility_60d','stock_momentum_12m','credit_spread_sector',
                       'interest_rate_sensitivity','market_cap_quintile']

    nlp_features = ['earnings_sentiment_score','tone_shift_qoq','risk_topic_regulatory',
                    'risk_topic_operational','risk_topic_financial','filing_text_complexity']

    # Make market/NLP features meaningfully correlated with default
    for col in market_features:
        base = rng.normal(0, 1.0, n_samples)
        df[col] = base + 0.8 * df["default"]

    for col in nlp_features:
        base = rng.normal(0, 1.0, n_samples)
        df[col] = base + 1.2 * df["default"]

    return df, fundamental_features, market_features, nlp_features

# --- Base Model Training Function ---
def train_base_models(X_train, y_train, fundamental_features, market_features, nlp_features, random_state=42):
    """
    Trains individual base models on their respective feature sets.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        fundamental_features (list): List of fundamental feature names.
        market_features (list): List of market feature names.
        nlp_features (list): List of NLP feature names.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing trained base models.
    """
    trained_models = {}

    # 1. Logistic Regression on Fundamental Features
    pipeline_fund = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1, random_state=random_state))
    ])
    pipeline_fund.fit(X_train[fundamental_features], y_train)
    trained_models['fund'] = pipeline_fund

    # 2. XGBoost on Market Features
    model_mkt = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              eval_metric='auc', random_state=random_state, enable_categorical=False)
    model_mkt.fit(X_train[market_features], y_train)
    trained_models['mkt'] = model_mkt

    # 3. LightGBM on NLP Features
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight_nlp = neg / pos if pos > 0 else 1.0 # Handle case with no positive samples

    model_nlp = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_nlp, random_state=random_state, n_jobs=-1
    )
    model_nlp.fit(X_train[nlp_features], y_train)
    trained_models['nlp'] = model_nlp

    return trained_models

# --- Prediction Function for Base Models ---
def predict_base_models(trained_models, X_test, fundamental_features, market_features, nlp_features):
    """
    Generates probability predictions from trained base models.

    Args:
        trained_models (dict): Dictionary of trained base models.
        X_test (pd.DataFrame): Test features.
        fundamental_features (list): List of fundamental feature names.
        market_features (list): List of market feature names.
        nlp_features (list): List of NLP feature names.

    Returns:
        tuple: A tuple containing:
            - np.array: Probabilities from the fundamental model.
            - np.array: Probabilities from the market model.
            - np.array: Probabilities from the NLP model.
    """
    prob_fund = trained_models['fund'].predict_proba(X_test[fundamental_features])[:, 1]
    prob_mkt = trained_models['mkt'].predict_proba(X_test[market_features])[:, 1]
    prob_nlp = trained_models['nlp'].predict_proba(X_test[nlp_features])[:, 1]
    return prob_fund, prob_mkt, prob_nlp

# --- Evaluation Function for Base Models ---
def evaluate_base_models(y_test, prob_fund, prob_mkt, prob_nlp):
    """
    Calculates AUC scores for individual base models.

    Args:
        y_test (pd.Series): True target values for the test set.
        prob_fund (np.array): Predicted probabilities from the fundamental model.
        prob_mkt (np.array): Predicted probabilities from the market model.
        prob_nlp (np.array): Predicted probabilities from the NLP model.

    Returns:
        tuple: A tuple containing AUC scores: (auc_fund, auc_mkt, auc_nlp).
    """
    auc_fund = roc_auc_score(y_test, prob_fund)
    auc_mkt = roc_auc_score(y_test, prob_mkt)
    auc_nlp = roc_auc_score(y_test, prob_nlp)
    return auc_fund, auc_mkt, auc_nlp

# --- Ensemble Model Training Function ---
def train_ensemble_models(X_train, y_train, fundamental_features, market_features, nlp_features, random_state=42):
    """
    Trains Voting and Stacking ensemble models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        fundamental_features (list): List of fundamental feature names.
        market_features (list): List of market feature names.
        nlp_features (list): List of NLP feature names.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: A tuple containing the fitted VotingClassifier and StackingClassifier.
    """
    # Base estimators for ensembles
    # Note: these are UNFITTED estimators that the ensemble will clone and fit
    
    # Calculate scale_pos_weight for LGBM in ensembles
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight_lgbm_ensemble = neg / pos if pos > 0 else 1.0

    base_pipeline_fund = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1, random_state=random_state))
    ])

    base_pipeline_mkt = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="auc", random_state=random_state, enable_categorical=False
    )

    base_pipeline_nlp = LGBMClassifier(
        n_estimators=200, learning_rate=0.1, num_leaves=31, max_depth=-1,
        scale_pos_weight=scale_pos_weight_lgbm_ensemble,
        random_state=random_state, n_jobs=-1
    )

    # Voting Classifier
    voting_soft = VotingClassifier(
        estimators=[
            ("fund", FeatureSubsetClassifier(base_pipeline_fund, fundamental_features)),
            ("mkt", FeatureSubsetClassifier(base_pipeline_mkt, market_features)),
            ("nlp", FeatureSubsetClassifier(base_pipeline_nlp, nlp_features)),
        ],
        voting="soft",
        weights=[0.3, 0.4, 0.3],
        n_jobs=-1
    )
    voting_soft.fit(X_train, y_train)

    # Stacking Classifier
    stacking = StackingClassifier(
        estimators=[
            ("fund", FeatureSubsetClassifier(base_pipeline_fund, fundamental_features)),
            ("mkt", FeatureSubsetClassifier(base_pipeline_mkt, market_features)),
            ("nlp", FeatureSubsetClassifier(base_pipeline_nlp, nlp_features)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1
    )
    stacking.fit(X_train, y_train)

    return voting_soft, stacking

# --- Prediction Function for Ensemble Models ---
def predict_ensemble_models(voting_soft, stacking, X_test):
    """
    Generates probability predictions from trained ensemble models.

    Args:
        voting_soft (VotingClassifier): Fitted Voting Classifier.
        stacking (StackingClassifier): Fitted Stacking Classifier.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple: A tuple containing:
            - np.array: Probabilities from the Voting ensemble.
            - np.array: Probabilities from the Stacking ensemble.
    """
    prob_vote = voting_soft.predict_proba(X_test)[:, 1]
    prob_stack = stacking.predict_proba(X_test)[:, 1]
    return prob_vote, prob_stack

# --- Evaluation Function for Ensemble Models ---
def evaluate_ensemble_models(y_test, prob_vote, prob_stack):
    """
    Calculates AUC scores for ensemble models.

    Args:
        y_test (pd.Series): True target values for the test set.
        prob_vote (np.array): Predicted probabilities from the Voting ensemble.
        prob_stack (np.array): Predicted probabilities from the Stacking ensemble.

    Returns:
        tuple: A tuple containing AUC scores: (auc_vote, auc_stack).
    """
    auc_vote = roc_auc_score(y_test, prob_vote)
    auc_stack = roc_auc_score(y_test, prob_stack)
    return auc_vote, auc_stack

# --- Calculate Average Ensemble AUC ---
def calculate_average_ensemble_auc(y_test, prob_fund, prob_mkt, prob_nlp):
    """
    Calculates AUC for a simple average of base model probabilities.

    Args:
        y_test (pd.Series): True target values for the test set.
        prob_fund (np.array): Probabilities from the fundamental model.
        prob_mkt (np.array): Probabilities from the market model.
        prob_nlp (np.array): Probabilities from the NLP model.

    Returns:
        tuple: A tuple containing:
            - np.array: Averaged probabilities.
            - float: AUC score for the average ensemble.
    """
    prob_avg = (prob_fund + prob_mkt + prob_nlp) / 3
    auc_avg = roc_auc_score(y_test, prob_avg)
    return prob_avg, auc_avg

# --- Bootstrap AUC Function ---
def bootstrap_auc(y_true, y_prob, n_boot=1000, random_state=42):
    """
    Performs bootstrap resampling to estimate the confidence interval of AUC.

    Args:
        y_true (pd.Series): True target values.
        y_prob (np.array): Predicted probabilities.
        n_boot (int): Number of bootstrap samples.
        random_state (int): Seed for reproducibility.

    Returns:
        list: A list containing the 2.5th and 97.5th percentiles (95% CI) of AUCs.
    """
    aucs = []
    rng = np.random.default_rng(random_state)
    for _ in range(n_boot):
        idx = rng.choice(range(len(y_true)), size=len(y_true), replace=True)
        # Ensure there are both positive and negative samples in the bootstrap sample
        if len(np.unique(y_true.iloc[idx])) > 1:
            aucs.append(roc_auc_score(y_true.iloc[idx], y_prob[idx]))
        else:
            # If only one class, AUC is undefined, skip
            continue
    if not aucs:
        return [np.nan, np.nan] # Return NaN if no valid AUCs were computed
    return np.percentile(aucs, [2.5, 97.5])

# --- Print All Results Function ---
def print_all_results(y_test, prob_fund, prob_mkt, prob_nlp, prob_avg, prob_vote, prob_stack,
                      auc_fund, auc_mkt, auc_nlp, auc_avg, auc_vote, auc_stack, random_state=42):
    """
    Prints all calculated AUC scores and bootstrap confidence interval for Stacking.

    Args:
        y_test (pd.Series): True target values for the test set.
        prob_fund (np.array): Predicted probabilities from fundamental model.
        prob_mkt (np.array): Predicted probabilities from market model.
        prob_nlp (np.array): Predicted probabilities from NLP model.
        prob_avg (np.array): Predicted probabilities from average ensemble.
        prob_vote (np.array): Predicted probabilities from Voting ensemble.
        prob_stack (np.array): Predicted probabilities from Stacking ensemble.
        auc_fund (float): AUC for fundamental model.
        auc_mkt (float): AUC for market model.
        auc_nlp (float): AUC for NLP model.
        auc_avg (float): AUC for average ensemble.
        auc_vote (float): AUC for Voting ensemble.
        auc_stack (float): AUC for Stacking ensemble.
        random_state (int): Seed for reproducibility.
    """
    print("\n--- Model Performance (AUC Scores) ---")
    print(f"Fundamental Model: {auc_fund:.4f}")
    print(f"Market Signal Model: {auc_mkt:.4f}")
    print(f"NLP Model: {auc_nlp:.4f}")
    print(f"Average Ensemble: {auc_avg:.4f}")
    print(f"Voting Ensemble: {auc_vote:.4f}")
    print(f"Stacking Ensemble: {auc_stack:.4f}")

    ci_stack = bootstrap_auc(y_test, prob_stack, random_state=random_state)
    print(f"Stacking Ensemble 95% CI: [{ci_stack[0]:.4f}, {ci_stack[1]:.4f}]")

# --- Agreement Analysis Function ---
def perform_agreement_analysis(prob_fund, prob_mkt, prob_nlp, y_test, prob_stack, threshold=0.15):
    """
    Analyzes the agreement between base models and its correlation with true default rates.

    Args:
        prob_fund (np.array): Predicted probabilities from fundamental model.
        prob_mkt (np.array): Predicted probabilities from market model.
        prob_nlp (np.array): Predicted probabilities from NLP model.
        y_test (pd.Series): True target values for the test set.
        prob_stack (np.array): Predicted probabilities from Stacking ensemble.
        threshold (float): Probability threshold to classify as default.
    """
    pred_fund = (prob_fund > threshold).astype(int)
    pred_mkt = (prob_mkt > threshold).astype(int)
    pred_nlp = (prob_nlp > threshold).astype(int)

    agreement = pred_fund + pred_mkt + pred_nlp

    confidence_df = pd.DataFrame({'agreement': agreement, 'true_default': y_test.values, 'prob_stack': prob_stack})
    print("\n--- Default Rate by Model Agreement ---")
    for n_agree in [0, 1, 2, 3]:  # 0, 1, 2, or 3 models flagging a default
        mask = confidence_df['agreement'] == n_agree
        if mask.sum() > 0:
            default_rate = confidence_df.loc[mask, 'true_default'].mean()
            print(f"{n_agree}/3 models flag default: {mask.sum()} cases, actual default rate = {default_rate:.2%}")
        else:
            print(f"{n_agree}/3 models flag default: 0 cases")

# --- Main Execution Function ---
def run_credit_risk_analysis(n_samples=10000, test_size=0.2, random_state=42, agreement_threshold=0.15):
    """
    Orchestrates the entire credit risk analysis workflow:
    data simulation, model training, evaluation, and agreement analysis.

    Args:
        n_samples (int): Number of samples for data simulation.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility across all steps.
        agreement_threshold (float): Probability threshold for agreement analysis.

    Returns:
        dict: A dictionary containing the fitted StackingClassifier and feature lists,
              suitable for deployment in an app.
    """
    print(f"Starting Credit Risk Analysis with random_state={random_state}...")

    # 1. Simulate Data
    df, fundamental_features, market_features, nlp_features = simulate_credit_data(n_samples, random_state)
    print(f"Data simulated with {n_samples} samples.")

    X = df[fundamental_features + market_features + nlp_features]
    y = df['default']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Data split into train ({len(X_train)} samples) and test ({len(X_test)} samples).")
    print(f"Train default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")

    # 3. Train Base Models
    trained_base_models = train_base_models(X_train, y_train, fundamental_features, market_features, nlp_features, random_state)
    print("Base models (Fundamental, Market, NLP) trained.")

    # 4. Predict with Base Models
    prob_fund, prob_mkt, prob_nlp = predict_base_models(trained_base_models, X_test, fundamental_features, market_features, nlp_features)
    print("Predictions made by base models.")

    # 5. Evaluate Base Models
    auc_fund, auc_mkt, auc_nlp = evaluate_base_models(y_test, prob_fund, prob_mkt, prob_nlp)
    print("Base models evaluated.")

    # 6. Train Ensemble Models
    voting_soft, stacking = train_ensemble_models(X_train, y_train, fundamental_features, market_features, nlp_features, random_state)
    print("Ensemble models (Voting, Stacking) trained.")

    # 7. Predict with Ensemble Models
    prob_vote, prob_stack = predict_ensemble_models(voting_soft, stacking, X_test)
    print("Predictions made by ensemble models.")

    # 8. Evaluate Ensemble Models
    auc_vote, auc_stack = evaluate_ensemble_models(y_test, prob_vote, prob_stack)
    print("Ensemble models evaluated.")

    # 9. Calculate Average Ensemble AUC
    prob_avg, auc_avg = calculate_average_ensemble_auc(y_test, prob_fund, prob_mkt, prob_nlp)
    print("Average ensemble calculated.")

    # 10. Print All Results
    print_all_results(y_test, prob_fund, prob_mkt, prob_nlp, prob_avg, prob_vote, prob_stack,
                      auc_fund, auc_mkt, auc_nlp, auc_avg, auc_vote, auc_stack, random_state)

    # 11. Perform Agreement Analysis
    perform_agreement_analysis(prob_fund, prob_mkt, prob_nlp, y_test, prob_stack, agreement_threshold)

    print("\nCredit Risk Analysis complete.")

    # Return key artifacts for potential use in an app.py
    return {
        "stacking_model": stacking,
        "fundamental_features": fundamental_features,
        "market_features": market_features,
        "nlp_features": nlp_features,
        "all_features": fundamental_features + market_features + nlp_features,
        "X_test": X_test,
        "y_test": y_test,
        "auc_results": {
            "fund": auc_fund,
            "mkt": auc_mkt,
            "nlp": auc_nlp,
            "avg": auc_avg,
            "vote": auc_vote,
            "stack": auc_stack
        }
    }

# --- Main Guard for Script Execution ---
if __name__ == "__main__":
    # Example usage when this script is run directly
    analysis_results = run_credit_risk_analysis(n_samples=10000, test_size=0.2, random_state=42)

    # Example of how you might use the returned model in an app.py:
    # deployed_model = analysis_results["stacking_model"]
    # all_features = analysis_results["all_features"]

    # # Simulate new data for prediction
    # new_data_sample = analysis_results["X_test"].head(5) 
    # print("\n--- Example Prediction for New Data ---")
    # print("Input features for first 5 samples:")
    # print(new_data_sample)
    
    # predictions = deployed_model.predict_proba(new_data_sample[all_features])[:, 1]
    # print("\nPredicted probabilities of default (first 5 samples):")
    # print(predictions)
    
    # # You could also save/load the model
    # # import joblib
    # # joblib.dump(deployed_model, 'credit_risk_stacking_model.pkl')
    # # loaded_model = joblib.load('credit_risk_stacking_model.pkl')
    # # loaded_predictions = loaded_model.predict_proba(new_data_sample[all_features])[:, 1]
    # # print("\nLoaded model predictions (first 5 samples):")
    # # print(loaded_predictions)
