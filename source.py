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
import os
import cloudpickle

# --- Model Cache Utilities ---
# cloudpickle is used instead of joblib/pickle so that custom classes like
# FeatureSubsetClassifier can be serialised even when Streamlit re-executes
# the script and the class object identity changes between runs.

BASE_MODELS_CACHE_FILE = "model_cache/base_models.pkl"
ENSEMBLE_MODELS_CACHE_FILE = "model_cache/ensemble_models.pkl"


def base_models_cache_exists(cache_dir="model_cache"):
    """Returns True if the base models cache file exists."""
    return os.path.isfile(os.path.join(cache_dir, "base_models.pkl"))


def ensemble_models_cache_exists(cache_dir="model_cache"):
    """Returns True if the ensemble models cache file exists."""
    return os.path.isfile(os.path.join(cache_dir, "ensemble_models.pkl"))


def save_base_models_cache(trained_model_fund_pipeline, prob_fund,
                           trained_model_mkt, prob_mkt,
                           trained_model_nlp, prob_nlp,
                           cache_dir="model_cache"):
    """Saves base model objects and predicted probabilities to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "trained_model_fund_pipeline": trained_model_fund_pipeline,
        "prob_fund": prob_fund,
        "trained_model_mkt": trained_model_mkt,
        "prob_mkt": prob_mkt,
        "trained_model_nlp": trained_model_nlp,
        "prob_nlp": prob_nlp,
    }
    with open(os.path.join(cache_dir, "base_models.pkl"), "wb") as f:
        cloudpickle.dump(payload, f)


def load_base_models_cache(cache_dir="model_cache"):
    """Loads base model objects and predicted probabilities from disk.

    Returns:
        tuple: (trained_model_fund_pipeline, prob_fund,
                trained_model_mkt, prob_mkt,
                trained_model_nlp, prob_nlp)
    """
    with open(os.path.join(cache_dir, "base_models.pkl"), "rb") as f:
        payload = cloudpickle.load(f)
    return (
        payload["trained_model_fund_pipeline"],
        payload["prob_fund"],
        payload["trained_model_mkt"],
        payload["prob_mkt"],
        payload["trained_model_nlp"],
        payload["prob_nlp"],
    )


def save_ensemble_models_cache(prob_avg, prob_vote, prob_stack,
                               voting_soft_model, stacking_model,
                               meta_learner_coefs,
                               cache_dir="model_cache"):
    """Saves ensemble model objects and predicted probabilities to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "prob_avg": prob_avg,
        "prob_vote": prob_vote,
        "prob_stack": prob_stack,
        "voting_soft_model": voting_soft_model,
        "stacking_model": stacking_model,
        "meta_learner_coefs": meta_learner_coefs,
    }
    with open(os.path.join(cache_dir, "ensemble_models.pkl"), "wb") as f:
        cloudpickle.dump(payload, f)


def load_ensemble_models_cache(cache_dir="model_cache"):
    """Loads ensemble model objects and predicted probabilities from disk.

    Returns:
        tuple: (prob_avg, prob_vote, prob_stack,
                voting_soft_model, stacking_model, meta_learner_coefs)
    """
    with open(os.path.join(cache_dir, "ensemble_models.pkl"), "rb") as f:
        payload = cloudpickle.load(f)
    return (
        payload["prob_avg"],
        payload["prob_vote"],
        payload["prob_stack"],
        payload["voting_soft_model"],
        payload["stacking_model"],
        payload["meta_learner_coefs"],
    )


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
        # Pass deep=False to super so sklearn clone gets only the direct __init__ params
        # (estimator, features) and can reconstruct correctly.
        params = super().get_params(deep=False)
        if deep and hasattr(self.estimator, "get_params"):
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
            if key in valid_params:  # Check if it's a direct parameter of FeatureSubsetClassifier
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

    fundamental_features = ['fico_score', 'dti', 'income', 'loan_amount', 'ltv', 'delinquencies_2yr',
                            'open_accounts', 'revolving_utilization', 'employment_length', 'home_ownership_encoded']

    market_features = ['equity_volatility_60d', 'stock_momentum_12m', 'credit_spread_sector',
                       'interest_rate_sensitivity', 'market_cap_quintile']

    nlp_features = ['earnings_sentiment_score', 'tone_shift_qoq', 'risk_topic_regulatory',
                    'risk_topic_operational', 'risk_topic_financial', 'filing_text_complexity']

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
        ('model', LogisticRegression(class_weight='balanced',
         max_iter=1000, C=0.1, random_state=random_state))
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
    # Handle case with no positive samples
    scale_pos_weight_nlp = neg / pos if pos > 0 else 1.0

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
    prob_fund = trained_models['fund'].predict_proba(
        X_test[fundamental_features])[:, 1]
    prob_mkt = trained_models['mkt'].predict_proba(
        X_test[market_features])[:, 1]
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
        ("model", LogisticRegression(class_weight="balanced",
         max_iter=1000, C=0.1, random_state=random_state))
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
            ("fund", FeatureSubsetClassifier(
                base_pipeline_fund, fundamental_features)),
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
            ("fund", FeatureSubsetClassifier(
                base_pipeline_fund, fundamental_features)),
            ("mkt", FeatureSubsetClassifier(base_pipeline_mkt, market_features)),
            ("nlp", FeatureSubsetClassifier(base_pipeline_nlp, nlp_features)),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000, random_state=random_state),
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
        return [np.nan, np.nan]  # Return NaN if no valid AUCs were computed
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

# --- Data Preparation Function for App ---


def prepare_credit_data(n_samples=10000, test_size=0.2, random_state=42):
    """
    Prepares the credit dataset and initialises base estimator objects (unfitted).

    Args:
        n_samples (int): Number of samples to simulate.
        test_size (float): Proportion of data for the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, fundamental_features, market_features,
                nlp_features, df_raw, scaler_fund, fund_base_estimator,
                mkt_base_estimator, nlp_base_estimator)
    """
    df_raw, fundamental_features, market_features, nlp_features = simulate_credit_data(
        n_samples, random_state)

    X = df_raw[fundamental_features + market_features + nlp_features]
    y = df_raw['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler_fund = StandardScaler()

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    fund_base_estimator = LogisticRegression(
        class_weight='balanced', max_iter=1000, C=0.1, random_state=random_state
    )
    mkt_base_estimator = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric='auc', random_state=random_state, enable_categorical=False
    )
    nlp_base_estimator = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1
    )

    return (X_train, X_test, y_train, y_test,
            fundamental_features, market_features, nlp_features,
            df_raw, scaler_fund,
            fund_base_estimator, mkt_base_estimator, nlp_base_estimator)


# --- Individual Model Training Function for App ---
def train_individual_models(X_train, y_train, X_test,
                            fundamental_features, market_features, nlp_features,
                            scaler_fund, fund_base_estimator,
                            mkt_base_estimator, nlp_base_estimator):
    """
    Trains three individual base models (Fundamental, Market, NLP) on their respective feature sets.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        fundamental_features (list): Fundamental feature names.
        market_features (list): Market feature names.
        nlp_features (list): NLP feature names.
        scaler_fund (StandardScaler): Scaler instance to use in the fundamental pipeline.
        fund_base_estimator: Unfitted Logistic Regression estimator.
        mkt_base_estimator: Unfitted XGBoost estimator.
        nlp_base_estimator: Unfitted LightGBM estimator.

    Returns:
        tuple: (trained_model_fund_pipeline, prob_fund, auc_fund,
                trained_model_mkt, prob_mkt, auc_mkt,
                trained_model_nlp, prob_nlp, auc_nlp)
                auc values are None as y_test labels are not available here.
    """
    from sklearn.base import clone

    pipeline_fund = Pipeline([
        ('scaler', clone(scaler_fund)),
        ('model', clone(fund_base_estimator))
    ])
    pipeline_fund.fit(X_train[fundamental_features], y_train)

    model_mkt = clone(mkt_base_estimator)
    model_mkt.fit(X_train[market_features], y_train)

    model_nlp = clone(nlp_base_estimator)
    model_nlp.fit(X_train[nlp_features], y_train)

    prob_fund = pipeline_fund.predict_proba(X_test[fundamental_features])[:, 1]
    prob_mkt = model_mkt.predict_proba(X_test[market_features])[:, 1]
    prob_nlp = model_nlp.predict_proba(X_test[nlp_features])[:, 1]

    return (pipeline_fund, prob_fund, None,
            model_mkt, prob_mkt, None,
            model_nlp, prob_nlp, None)


# --- Prediction Correlation Heatmap ---
def plot_prediction_correlation_heatmap(prob_fund, prob_mkt, prob_nlp):
    """
    Plots a heatmap of pairwise Pearson correlations between base-model probability predictions.

    Args:
        prob_fund (np.array): Probabilities from the fundamental model.
        prob_mkt (np.array): Probabilities from the market model.
        prob_nlp (np.array): Probabilities from the NLP model.

    Returns:
        matplotlib.figure.Figure: The heatmap figure.
    """
    pred_df = pd.DataFrame({
        'Fundamental': prob_fund,
        'Market Signal': prob_mkt,
        'NLP': prob_nlp
    })
    corr = pred_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title('Base Model Prediction Correlation Matrix',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# --- Ensemble Strategy Training Function for App ---
def train_ensemble_strategies(X_train, y_train, X_test, y_test,
                              prob_fund, prob_mkt, prob_nlp,
                              fundamental_features, market_features, nlp_features,
                              fund_base_estimator, mkt_base_estimator, nlp_base_estimator,
                              trained_model_fund_pipeline):
    """
    Trains three ensemble strategies: simple probability averaging, soft-voting, and stacking.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target (for AUC calculation).
        prob_fund (np.array): Test probabilities from the fundamental model.
        prob_mkt (np.array): Test probabilities from the market model.
        prob_nlp (np.array): Test probabilities from the NLP model.
        fundamental_features (list): Fundamental feature names.
        market_features (list): Market feature names.
        nlp_features (list): NLP feature names.
        fund_base_estimator: Unfitted base LR estimator.
        mkt_base_estimator: Unfitted base XGB estimator.
        nlp_base_estimator: Unfitted base LGBM estimator.
        trained_model_fund_pipeline: Already-fitted fundamental pipeline (unused directly; kept for API consistency).

    Returns:
        tuple: (prob_avg, auc_avg, prob_vote, auc_vote, prob_stack, auc_stack,
                voting_soft_model, stacking_model, meta_learner_coefs)
    """
    from sklearn.base import clone

    # 1. Average Ensemble
    prob_avg = (prob_fund + prob_mkt + prob_nlp) / 3.0
    auc_avg = roc_auc_score(y_test, prob_avg)

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    base_pipeline_fund = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clone(fund_base_estimator))
    ])
    base_model_mkt = clone(mkt_base_estimator)
    base_model_nlp = LGBMClassifier(
        n_estimators=200, learning_rate=0.1, num_leaves=31, max_depth=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1
    )

    # 2. Soft Voting Ensemble
    voting_soft_model = VotingClassifier(
        estimators=[
            ('fund', FeatureSubsetClassifier(
                base_pipeline_fund, fundamental_features)),
            ('mkt', FeatureSubsetClassifier(clone(base_model_mkt), market_features)),
            ('nlp', FeatureSubsetClassifier(base_model_nlp, nlp_features)),
        ],
        voting='soft',
        weights=[0.3, 0.4, 0.3],
        n_jobs=-1
    )
    voting_soft_model.fit(X_train, y_train)
    prob_vote = voting_soft_model.predict_proba(X_test)[:, 1]
    auc_vote = roc_auc_score(y_test, prob_vote)

    # 3. Stacking Ensemble
    stacking_model = StackingClassifier(
        estimators=[
            ('fund', FeatureSubsetClassifier(Pipeline([('scaler', StandardScaler(
            )), ('model', clone(fund_base_estimator))]), fundamental_features)),
            ('mkt', FeatureSubsetClassifier(
                clone(mkt_base_estimator), market_features)),
            ('nlp', FeatureSubsetClassifier(LGBMClassifier(
                n_estimators=200, learning_rate=0.1, num_leaves=31, max_depth=-1,
                scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
            ), nlp_features)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )
    stacking_model.fit(X_train, y_train)
    prob_stack = stacking_model.predict_proba(X_test)[:, 1]
    auc_stack = roc_auc_score(y_test, prob_stack)

    # Extract meta-learner coefficients
    meta_learner_coefs = stacking_model.final_estimator_.coef_[0]

    return (prob_avg, auc_avg,
            prob_vote, auc_vote,
            prob_stack, auc_stack,
            voting_soft_model, stacking_model, meta_learner_coefs)


# --- Meta-Learner Weights Plot ---
def plot_meta_learner_weights(meta_learner_coefs):
    """
    Plots the coefficients of the meta-learner in the Stacking ensemble.

    Args:
        meta_learner_coefs (np.array): Coefficients from the logistic regression meta-learner.

    Returns:
        matplotlib.figure.Figure: Bar chart figure.
    """
    model_names = ['Fundamental\nModel', 'Market Signal\nModel', 'NLP\nModel']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(model_names, meta_learner_coefs, color=colors,
                  edgecolor='black', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Stacking Ensemble: Meta-Learner Coefficients',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=11)
    ax.set_xlabel('Base Model', fontsize=11)
    for bar, coef in zip(bars, meta_learner_coefs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{coef:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return fig


# --- Bootstrap Confidence Intervals for All Models ---
def calculate_bootstrap_cis(y_test, all_probabilities_dict, n_boot=1000, random_state=42):
    """
    Calculates bootstrap confidence intervals for all models in the dictionary.

    Args:
        y_test (pd.Series): True target values.
        all_probabilities_dict (dict): Dict mapping model name to predicted probabilities array.
        n_boot (int): Number of bootstrap samples.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns ['Model', 'Bootstrap AUC'].
                      Used for violin plots and CI computation.
    """
    records = []
    rng = np.random.default_rng(random_state)
    y_arr = np.array(y_test)

    for model_name, probs in all_probabilities_dict.items():
        for _ in range(n_boot):
            idx = rng.choice(len(y_arr), size=len(y_arr), replace=True)
            if len(np.unique(y_arr[idx])) > 1:
                auc_val = roc_auc_score(y_arr[idx], probs[idx])
                records.append({'Model': model_name, 'Bootstrap AUC': auc_val})

    return pd.DataFrame(records)


# --- Model Summary DataFrame ---
def prepare_model_summary_df(all_auc_scores_dict, ci_results_df):
    """
    Prepares a summary DataFrame combining AUC scores and bootstrap confidence intervals.

    Args:
        all_auc_scores_dict (dict): Dict mapping model name to AUC score.
        ci_results_df (pd.DataFrame): Long-format bootstrap CI DataFrame from calculate_bootstrap_cis.

    Returns:
        pd.DataFrame: Summary DataFrame with columns: Model, AUC, AUC Lift, 95% CI Lower,
                      95% CI Upper, CI Width.
    """
    base_models = ['Fundamental Model', 'Market Signal Model', 'NLP Model']
    base_aucs = [all_auc_scores_dict[m]
                 for m in base_models if m in all_auc_scores_dict]
    best_base_auc = max(base_aucs) if base_aucs else 0.5

    ci_summary = (
        ci_results_df.groupby('Model')['Bootstrap AUC']
        .quantile([0.025, 0.975])
        .unstack(level=-1)
        .rename(columns={0.025: '95% CI Lower', 0.975: '95% CI Upper'})
    )
    ci_summary['CI Width'] = ci_summary['95% CI Upper'] - \
        ci_summary['95% CI Lower']

    rows = []
    for model_name, auc_val in all_auc_scores_dict.items():
        row = {'Model': model_name, 'AUC': auc_val,
               'AUC Lift': auc_val - best_base_auc}
        if model_name in ci_summary.index:
            row['95% CI Lower'] = ci_summary.loc[model_name, '95% CI Lower']
            row['95% CI Upper'] = ci_summary.loc[model_name, '95% CI Upper']
            row['CI Width'] = ci_summary.loc[model_name, 'CI Width']
        else:
            row['95% CI Lower'] = np.nan
            row['95% CI Upper'] = np.nan
            row['CI Width'] = np.nan
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index('Model')
    return summary_df


# --- AUC Bar Chart ---
def plot_auc_bar_chart(results_df_summary):
    """
    Plots a bar chart of AUC scores for all models.

    Args:
        results_df_summary (pd.DataFrame): Summary DataFrame from prepare_model_summary_df.

    Returns:
        matplotlib.figure.Figure: Bar chart figure.
    """
    base_colors = ['#5C85D6', '#5CAD6E', '#E07B39']
    ensemble_colors = ['#9C59CC', '#CC5959', '#CC9920']
    model_names = list(results_df_summary.index)
    aucs = results_df_summary['AUC'].values

    colors = []
    for name in model_names:
        if 'Ensemble' in name:
            colors.append(ensemble_colors[len(colors) % len(ensemble_colors)])
        else:
            colors.append(base_colors[len(colors) % len(base_colors)])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, aucs, color=colors,
                  edgecolor='black', linewidth=0.8)
    ax.set_ylim(max(0.5, float(np.min(aucs)) - 0.02),
                min(1.0, float(np.max(aucs)) + 0.04))
    ax.set_title('AUC Comparison: Base Models vs Ensemble Strategies',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    plt.xticks(rotation=20, ha='right')
    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{auc_val:.4f}', ha='center', va='bottom', fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5C85D6', label='Base Models'),
        Patch(facecolor='#9C59CC', label='Ensemble Models')
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    return fig


# --- ROC Curves Overlay ---
def plot_roc_curves_overlay(y_test, all_probabilities_dict):
    """
    Plots ROC curves for all models on a single axes.

    Args:
        y_test (pd.Series): True target values.
        all_probabilities_dict (dict): Dict mapping model name to predicted probabilities.

    Returns:
        matplotlib.figure.Figure: ROC overlay figure.
    """
    from sklearn.metrics import roc_curve

    base_style = {'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.75}
    ensemble_style = {'linestyle': '-', 'linewidth': 2.2, 'alpha': 0.95}

    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, probs in all_probabilities_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        style = ensemble_style if 'Ensemble' in model_name else base_style
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_val:.4f})', **style)

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1,
            label='Random Classifier (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# --- Bootstrap Violin Plot ---
def plot_bootstrap_violin_plot(ci_results_df):
    """
    Plots a violin plot of bootstrap AUC distributions for all models.

    Args:
        ci_results_df (pd.DataFrame): Long-format DataFrame with columns ['Model', 'Bootstrap AUC']
                                      from calculate_bootstrap_cis.

    Returns:
        matplotlib.figure.Figure: Violin plot figure.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    model_order = [m for m in [
        'Fundamental Model', 'Market Signal Model', 'NLP Model',
        'Average Ensemble', 'Voting Ensemble', 'Stacking Ensemble'
    ] if m in ci_results_df['Model'].unique()]

    palette = {m: ('#5C85D6' if 'Ensemble' not in m else '#9C59CC')
               for m in model_order}

    sns.violinplot(data=ci_results_df, x='Model', y='Bootstrap AUC',
                   order=model_order, hue='Model', palette=palette,
                   legend=False,
                   inner='box', cut=0, ax=ax)
    ax.tick_params(axis='x', rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')
    ax.set_title('Bootstrap AUC Distribution (1,000 resamples)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_xlabel('')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# --- Agreement Analysis Function ---
def perform_agreement_analysis(y_test, prob_fund, prob_mkt, prob_nlp, prob_stack,
                               threshold=0.15, X_test=None,
                               fundamental_features=None, market_features=None,
                               nlp_features=None):
    """
    Analyses the agreement between base models and its correlation with true default rates.
    Identifies disagreement cases for analyst review.

    Args:
        y_test (pd.Series): True target values for the test set.
        prob_fund (np.array): Predicted probabilities from fundamental model.
        prob_mkt (np.array): Predicted probabilities from market model.
        prob_nlp (np.array): Predicted probabilities from NLP model.
        prob_stack (np.array): Predicted probabilities from Stacking ensemble.
        threshold (float): Probability threshold to classify as default.
        X_test (pd.DataFrame, optional): Test features (used for disagreement case details).
        fundamental_features (list, optional): Fundamental feature names.
        market_features (list, optional): Market feature names.
        nlp_features (list, optional): NLP feature names.

    Returns:
        tuple: (confidence_df, disagreement_cases_df)
            - confidence_df (pd.DataFrame): Per-sample agreement level and true default.
            - disagreement_cases_df (pd.DataFrame): Subset of cases where models disagree (1 or 2 out of 3 flag default).
    """
    pred_fund = (prob_fund > threshold).astype(int)
    pred_mkt = (prob_mkt > threshold).astype(int)
    pred_nlp = (prob_nlp > threshold).astype(int)

    agreement = pred_fund + pred_mkt + pred_nlp

    confidence_df = pd.DataFrame({
        'agreement': agreement,
        'true_default': np.array(y_test),
        'prob_stack': prob_stack,
        'prob_fund': prob_fund,
        'prob_mkt': prob_mkt,
        'prob_nlp': prob_nlp,
        'pred_fund': pred_fund,
        'pred_mkt': pred_mkt,
        'pred_nlp': pred_nlp,
    })

    # Disagreement cases: exactly 1 or 2 out of 3 models flag default
    disagree_mask = confidence_df['agreement'].isin([1, 2])
    disagreement_cases_df = confidence_df[disagree_mask].copy()
    disagreement_cases_df['agreement_label'] = disagreement_cases_df['agreement'].map(
        {1: '1/3 models flag default', 2: '2/3 models flag default'}
    )

    if X_test is not None:
        # Add a subset of features for analyst context
        feature_cols = []
        if fundamental_features:
            feature_cols += fundamental_features[:3]
        if market_features:
            feature_cols += market_features[:2]
        if nlp_features:
            feature_cols += nlp_features[:2]
        feature_cols = [c for c in feature_cols if c in X_test.columns]
        if feature_cols:
            disagreement_cases_df = disagreement_cases_df.join(
                X_test[feature_cols].reset_index(drop=True)
                if disagreement_cases_df.index.max() >= len(X_test)
                else X_test[feature_cols]
            )

    disagreement_cases_df = disagreement_cases_df.head(
        50).reset_index(drop=True)
    return confidence_df, disagreement_cases_df


# --- Model Agreement Stacked Bar Chart ---
def plot_model_agreement_stacked_bar(confidence_df):
    """
    Plots a stacked bar chart showing case distribution and actual default rate
    by level of model agreement.

    Args:
        confidence_df (pd.DataFrame): DataFrame from perform_agreement_analysis with
                                      'agreement' and 'true_default' columns.

    Returns:
        matplotlib.figure.Figure: Stacked bar figure.
    """
    agreement_levels = [0, 1, 2, 3]
    labels = ['0/3 (Green)', '1/3 (Yellow)', '2/3 (Yellow)', '3/3 (Red)']
    triage_colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']

    counts = []
    default_rates = []
    for lvl in agreement_levels:
        mask = confidence_df['agreement'] == lvl
        n = mask.sum()
        counts.append(n)
        default_rates.append(
            confidence_df.loc[mask, 'true_default'].mean() if n > 0 else 0.0)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    bars = ax1.bar(labels, counts, color=triage_colors,
                   edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Number of Cases', fontsize=11)
    ax1.set_title('Model Agreement Distribution & Actual Default Rate',
                  fontsize=13, fontweight='bold')
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(cnt), ha='center', va='bottom', fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(labels, [r * 100 for r in default_rates], 'ko--',
             linewidth=2, markersize=7, label='Actual Default Rate (%)')
    ax2.set_ylabel('Actual Default Rate (%)', fontsize=11)
    ax2.set_ylim(0, max(default_rates) * 100 * 1.4 + 5)
    ax2.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    return fig

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
    df, fundamental_features, market_features, nlp_features = simulate_credit_data(
        n_samples, random_state)
    print(f"Data simulated with {n_samples} samples.")

    X = df[fundamental_features + market_features + nlp_features]
    y = df['default']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(
        f"Data split into train ({len(X_train)} samples) and test ({len(X_test)} samples).")
    print(f"Train default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")

    # 3. Train Base Models
    trained_base_models = train_base_models(
        X_train, y_train, fundamental_features, market_features, nlp_features, random_state)
    print("Base models (Fundamental, Market, NLP) trained.")

    # 4. Predict with Base Models
    prob_fund, prob_mkt, prob_nlp = predict_base_models(
        trained_base_models, X_test, fundamental_features, market_features, nlp_features)
    print("Predictions made by base models.")

    # 5. Evaluate Base Models
    auc_fund, auc_mkt, auc_nlp = evaluate_base_models(
        y_test, prob_fund, prob_mkt, prob_nlp)
    print("Base models evaluated.")

    # 6. Train Ensemble Models
    voting_soft, stacking = train_ensemble_models(
        X_train, y_train, fundamental_features, market_features, nlp_features, random_state)
    print("Ensemble models (Voting, Stacking) trained.")

    # 7. Predict with Ensemble Models
    prob_vote, prob_stack = predict_ensemble_models(
        voting_soft, stacking, X_test)
    print("Predictions made by ensemble models.")

    # 8. Evaluate Ensemble Models
    auc_vote, auc_stack = evaluate_ensemble_models(
        y_test, prob_vote, prob_stack)
    print("Ensemble models evaluated.")

    # 9. Calculate Average Ensemble AUC
    prob_avg, auc_avg = calculate_average_ensemble_auc(
        y_test, prob_fund, prob_mkt, prob_nlp)
    print("Average ensemble calculated.")

    # 10. Print All Results
    print_all_results(y_test, prob_fund, prob_mkt, prob_nlp, prob_avg, prob_vote, prob_stack,
                      auc_fund, auc_mkt, auc_nlp, auc_avg, auc_vote, auc_stack, random_state)

    # 11. Perform Agreement Analysis
    perform_agreement_analysis(
        y_test, prob_fund, prob_mkt, prob_nlp, prob_stack, agreement_threshold)

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
    analysis_results = run_credit_risk_analysis(
        n_samples=10000, test_size=0.2, random_state=42)

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
