import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from source import *

st.set_page_config(page_title="QuLab: Lab 24: Ensemble Model for Risk Prediction", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 24: Ensemble Model for Risk Prediction")
st.divider()

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = "Introduction & Data Preparation"
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'fundamental_features' not in st.session_state:
    st.session_state.fundamental_features = None
if 'market_features' not in st.session_state:
    st.session_state.market_features = None
if 'nlp_features' not in st.session_state:
    st.session_state.nlp_features = None
if 'scaler_fund' not in st.session_state:
    st.session_state.scaler_fund = None
if 'fund_base_estimator' not in st.session_state:
    st.session_state.fund_base_estimator = None
if 'mkt_base_estimator' not in st.session_state:
    st.session_state.mkt_base_estimator = None
if 'nlp_base_estimator' not in st.session_state:
    st.session_state.nlp_base_estimator = None

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'trained_model_fund_pipeline' not in st.session_state:
    st.session_state.trained_model_fund_pipeline = None
if 'trained_model_mkt' not in st.session_state:
    st.session_state.trained_model_mkt = None
if 'trained_model_nlp' not in st.session_state:
    st.session_state.trained_model_nlp = None
if 'prob_fund' not in st.session_state:
    st.session_state.prob_fund = None
if 'prob_mkt' not in st.session_state:
    st.session_state.prob_mkt = None
if 'prob_nlp' not in st.session_state:
    st.session_state.prob_nlp = None

if 'ensembles_trained' not in st.session_state:
    st.session_state.ensembles_trained = False
if 'prob_avg' not in st.session_state:
    st.session_state.prob_avg = None
if 'prob_vote' not in st.session_state:
    st.session_state.prob_vote = None
if 'prob_stack' not in st.session_state:
    st.session_state.prob_stack = None
if 'voting_soft_model' not in st.session_state:
    st.session_state.voting_soft_model = None
if 'stacking_model' not in st.session_state:
    st.session_state.stacking_model = None
if 'meta_learner_coefs' not in st.session_state:
    st.session_state.meta_learner_coefs = None

if 'all_probabilities_dict' not in st.session_state:
    st.session_state.all_probabilities_dict = None
if 'all_auc_scores_dict' not in st.session_state:
    st.session_state.all_auc_scores_dict = None
if 'results_df_summary' not in st.session_state:
    st.session_state.results_df_summary = None
if 'ci_results_df' not in st.session_state:
    st.session_state.ci_results_df = None

if 'agreement_threshold' not in st.session_state:
    st.session_state.agreement_threshold = 0.15
if 'confidence_df' not in st.session_state:
    st.session_state.confidence_df = None
if 'disagreement_cases_df' not in st.session_state:
    st.session_state.disagreement_cases_df = None

# Navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    [
        "Introduction & Data Preparation",
        "Base Model Training",
        "Ensemble Strategies",
        "Performance & Stability Analysis",
        "Model Agreement & Triage"
    ],
    key="page"
)

# Page 1: Introduction & Data Preparation
if st.session_state.page == "Introduction & Data Preparation":
    st.title("💡 Ensemble Model for Robust Corporate Bond Default Prediction")
    st.markdown(f"As a **CFA Charterholder and Credit Analyst**, your role involves accurately assessing corporate bond default risk. No single model perfectly captures all facets of risk. This application guides you through building a robust solution by combining diverse models and data sources, a practice central to advanced risk management and compliance with regulatory frameworks like **SR 11-7**.")

    st.header("1. Data Preparation: The Foundation of Robust Models")
    st.markdown(f"The first step in any robust risk assessment is meticulous data preparation. We leverage three distinct feature sets to provide varied 'views' of default risk:")
    st.markdown(f"-   **Fundamental Features**: Traditional financial ratios and borrower characteristics (e.g., FICO, DTI, Income).")
    st.markdown(f"-   **Market Signal Features**: Dynamic market-derived indicators (e.g., equity volatility, credit spreads).")
    st.markdown(f"-   **NLP Features**: Insights extracted from textual data (e.g., earnings call sentiment, risk topic analysis from 10-K filings).")
    st.markdown(f"This multi-faceted data approach is crucial for building models whose errors are decorrelated, a prerequisite for effective ensembling, much like diversifying a portfolio with uncorrelated assets.")

    if st.button("Load and Prepare Data", help="Simulates and prepares the credit default dataset."):
        with st.spinner("Loading and preparing data..."):
            X_train, X_test, y_train, y_test, fundamental_features, market_features, nlp_features, df_raw, scaler_fund, fund_base_estimator, mkt_base_estimator, nlp_base_estimator = prepare_credit_data()

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.fundamental_features = fundamental_features
            st.session_state.market_features = market_features
            st.session_state.nlp_features = nlp_features
            st.session_state.df_raw = df_raw
            st.session_state.scaler_fund = scaler_fund
            st.session_state.fund_base_estimator = fund_base_estimator
            st.session_state.mkt_base_estimator = mkt_base_estimator
            st.session_state.nlp_base_estimator = nlp_base_estimator
            st.session_state.data_loaded = True
        st.success("Data loaded and prepared successfully!")

    if st.session_state.data_loaded:
        st.markdown(f"### Data Overview")
        st.dataframe(st.session_state.df_raw.head())
        st.markdown(f"Dataset shape: {st.session_state.df_raw.shape[0]:,} rows, {st.session_state.df_raw.shape[1]:} columns")
        st.markdown(f"Training set shape: {st.session_state.X_train.shape[0]:,} samples")
        st.markdown(f"Test set shape: {st.session_state.X_test.shape[0]:,} samples")
        st.markdown(f"**Fundamental Features ({len(st.session_state.fundamental_features)}):** {', '.join(st.session_state.fundamental_features[:5])}... ")
        st.markdown(f"**Market Signal Features ({len(st.session_state.market_features)}):** {', '.join(st.session_state.market_features[:3])}... ")
        st.markdown(f"**NLP Features ({len(st.session_state.nlp_features)}):** {', '.join(st.session_state.nlp_features[:3])}... ")

# Page 2: Base Model Training
elif st.session_state.page == "Base Model Training":
    st.title("2. Train Diverse Base Models")
    st.markdown(f"As an investment professional, you understand the value of diverse perspectives. We will now train three distinct base models, each specialized to capture different signals from our multi-faceted data. This approach is fundamental to reducing correlated errors across models, a core principle for effective ensembling.")
    st.markdown(f"")
    st.markdown(f"")
    st.markdown(f"**Model 1: Fundamental Model (Logistic Regression)**")
    st.markdown(f"Focuses on traditional financial health indicators. Logistic Regression provides interpretability for the core drivers of default, aligning with standard credit analysis practices.")
    st.markdown(f"")
    st.markdown(f"**Model 2: Market Signal Model (XGBoost)**")
    st.markdown(f"Captures non-linear relationships and interactions within dynamic market data. This model reflects how market sentiment and external factors influence default probabilities.")
    st.markdown(f"")
    st.markdown(f"**Model 3: NLP Model (LightGBM)**")
    st.markdown(f"Leverages textual information (e.g., sentiment, risk topics from filings) often missed by quantitative models. LightGBM efficiently handles high-dimensional NLP features, providing a qualitative edge to our prediction.")

    if st.session_state.data_loaded:
        if st.button("Train Base Models", help="Trains Logistic Regression, XGBoost, and LightGBM models."):
            with st.spinner("Training individual models..."):
                trained_model_fund_pipeline, prob_fund, auc_fund, \
                trained_model_mkt, prob_mkt, auc_mkt, \
                trained_model_nlp, prob_nlp, auc_nlp = train_individual_models(
                    st.session_state.X_train, st.session_state.y_train, st.session_state.X_test,
                    st.session_state.fundamental_features, st.session_state.market_features, st.session_state.nlp_features,
                    st.session_state.scaler_fund, st.session_state.fund_base_estimator,
                    st.session_state.mkt_base_estimator, st.session_state.nlp_base_estimator
                )

                st.session_state.trained_model_fund_pipeline = trained_model_fund_pipeline
                st.session_state.prob_fund = prob_fund
                st.session_state.trained_model_mkt = trained_model_mkt
                st.session_state.prob_mkt = prob_mkt
                st.session_state.trained_model_nlp = trained_model_nlp
                st.session_state.prob_nlp = prob_nlp
                st.session_state.models_trained = True

                st.session_state.all_probabilities_dict = {
                    'Fundamental Model': prob_fund,
                    'Market Signal Model': prob_mkt,
                    'NLP Model': prob_nlp
                }
                st.session_state.all_auc_scores_dict = {}
            st.success("Base models trained successfully!")

        if st.session_state.models_trained:
            st.markdown(f"### Base Model Performance (AUC on Test Set)")
            auc_scores_base = {
                'Fundamental Model': roc_auc_score(st.session_state.y_test, st.session_state.prob_fund),
                'Market Signal Model': roc_auc_score(st.session_state.y_test, st.session_state.prob_mkt),
                'NLP Model': roc_auc_score(st.session_state.y_test, st.session_state.prob_nlp)
            }
            st.session_state.all_auc_scores_dict = {**st.session_state.all_auc_scores_dict, **auc_scores_base}

            st.dataframe(pd.DataFrame([auc_scores_base]).T.rename(columns={0: 'AUC Score'}).style.format("{:.4f}"))

            st.markdown(f"### Prediction Correlation Heatmap (V3)")
            st.markdown(f"Understanding the correlation of predictions between base models is key to confirming diversity. Lower correlation indicates that models are making different types of errors, which is ideal for ensembling.")
            st.markdown(r"$$\rho_{ij} = \text{Corr}(p_i, p_j)$$")
            st.markdown(r"where $\rho_{ij}$ is the pairwise correlation between predictions of model $i$ and model $j$, and $p_i, p_j$ are the predicted probabilities of default from models $i$ and $j$.")
            st.markdown(f"Ideally, we want $\rho_{ij} < 0.7$ for all pairs to maximize ensemble benefits.")
            fig_corr = plot_prediction_correlation_heatmap(st.session_state.prob_fund, st.session_state.prob_mkt, st.session_state.prob_nlp)
            st.pyplot(fig_corr)
            st.markdown(f"**Financial Interpretation:** The low correlation between the Fundamental and NLP models suggests that text signals capture incremental information beyond traditional financial ratios. This justifies investing in NLP infrastructure for credit analysis, as it genuinely adds a new 'view' of risk.")
    else:
        st.info("Please load and prepare the data first on the 'Introduction & Data Preparation' page.")

# Page 3: Ensemble Strategies
elif st.session_state.page == "Ensemble Strategies":
    st.title("3. Implement Ensemble Strategies: Diversifying Your Models")
    st.markdown(f"Just as you diversify investment portfolios to mitigate risk, ensemble methods combine multiple models to reduce overall prediction error and increase stability. This directly aligns with **SR 11-7** model risk management guidance, treating each component model as a 'challenger' to others.")

    st.header("a. The Diversity Principle: Why Ensembles Work")
    st.markdown(f"The core benefit of ensembling comes from combining models with decorrelated errors. If models make different mistakes, their combined prediction is often superior.")
    st.markdown(r"$$E\left[\frac{1}{M} \sum_{m=1}^{M} E_m\right]^2 = \frac{1}{M}\bar{e} + \frac{M-1}{M}\bar{c}$$")
    st.markdown(r"where $E_m$ is the error of individual model $m$, $M$ is the number of models, $\bar{e}$ is the average individual model error, and $\bar{c}$ is the average pairwise covariance of errors. Lower $\bar{c}$ leads to lower ensemble error.")
    st.markdown(f"**Financial Analogy**: This principle is identical to portfolio diversification, where combining uncorrelated assets (models) reduces overall risk (error). The prediction correlation matrix (V3) plays the role of the return correlation matrix in investment portfolios.")

    st.header("b. Ensemble Methods Implemented")
    st.markdown(f"**1. Simple Probability Averaging:**")
    st.markdown(f"The simplest approach, where each base model's predicted probability of default is given equal weight. Effective when individual models are of comparable quality.")
    st.markdown(r"$$\hat{p}_{\text{avg}} = \frac{1}{M} \sum_{m=1}^{M} p_m$$")
    st.markdown(r"where $\hat{p}_{\text{avg}}$ is the average ensemble probability, $M$ is the number of base models, and $p_m$ is the predicted probability from model $m$.")

    st.markdown(f"**2. Voting Classifier (Soft Voting):**")
    st.markdown(f"Combines predictions by averaging the predicted probabilities from each base model, allowing for weighted contributions. This is more flexible than simple averaging.")
    st.markdown(r"$$\hat{p}_{\text{vote}} = \sum_{m=1}^{M} w_m p_m$$")
    st.markdown(r"where $w_m$ are weights assigned to each model's prediction, reflecting their perceived importance or performance.")

    st.markdown(f"**3. Stacking (Meta-Learner):**")
    st.markdown(f"A more sophisticated approach where the predictions of the base models become new features for a 'meta-learner' (e.g., Logistic Regression). The meta-learner learns the optimal way to combine base model predictions.")
    st.markdown(r"$$\hat{p}_{\text{stack}} = \sigma\left(\beta_0 + \sum_{m=1}^{M} \beta_m p_m^{(CV)}\right)$$")
    st.markdown(r"where $\hat{p}_{\text{stack}}$ is the stacked ensemble probability, $\sigma$ is the sigmoid function (for Logistic Regression meta-learner), $\beta_0$ is the intercept, $\beta_m$ are the coefficients (weights) learned by the meta-learner for each base model's cross-validated prediction $p_m^{(CV)}$. Cross-validation is crucial to prevent meta-learner overfitting on the base model predictions.")

    if st.session_state.models_trained:
        if st.button("Train Ensemble Models", help="Trains Averaging, Voting, and Stacking ensembles."):
            with st.spinner("Training ensemble models..."):
                prob_avg, auc_avg, \
                prob_vote, auc_vote, \
                prob_stack, auc_stack, \
                voting_soft_model, stacking_model, meta_learner_coefs = train_ensemble_strategies(
                    st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test,
                    st.session_state.prob_fund, st.session_state.prob_mkt, st.session_state.prob_nlp,
                    st.session_state.fundamental_features, st.session_state.market_features, st.session_state.nlp_features,
                    st.session_state.fund_base_estimator, st.session_state.mkt_base_estimator, st.session_state.nlp_base_estimator,
                    st.session_state.trained_model_fund_pipeline
                )

                st.session_state.prob_avg = prob_avg
                st.session_state.prob_vote = prob_vote
                st.session_state.prob_stack = prob_stack
                st.session_state.voting_soft_model = voting_soft_model
                st.session_state.stacking_model = stacking_model
                st.session_state.meta_learner_coefs = meta_learner_coefs
                st.session_state.ensembles_trained = True

                st.session_state.all_probabilities_dict['Average Ensemble'] = prob_avg
                st.session_state.all_probabilities_dict['Voting Ensemble'] = prob_vote
                st.session_state.all_probabilities_dict['Stacking Ensemble'] = prob_stack
            st.success("Ensemble models trained successfully!")

        if st.session_state.ensembles_trained:
            st.markdown(f"### Ensemble Model Performance (AUC on Test Set)")
            auc_scores_ensemble = {
                'Average Ensemble': roc_auc_score(st.session_state.y_test, st.session_state.prob_avg),
                'Voting Ensemble': roc_auc_score(st.session_state.y_test, st.session_state.prob_vote),
                'Stacking Ensemble': roc_auc_score(st.session_state.y_test, st.session_state.prob_stack)
            }
            st.session_state.all_auc_scores_dict = {**st.session_state.all_auc_scores_dict, **auc_scores_ensemble}

            st.dataframe(pd.DataFrame([auc_scores_ensemble]).T.rename(columns={0: 'AUC Score'}).style.format("{:.4f}"))

            st.markdown(f"### Meta-Learner Weights (V6)")
            st.markdown(f"For the Stacking ensemble, the meta-learner assigns weights to each base model's prediction. These weights quantify which 'view' of default risk is most informative, providing valuable insights for financial professionals.")
            st.markdown(f"**Financial Interpretation:** A positive weight indicates the meta-learner found that model's predictions incrementally useful. This helps answer questions like, 'Does the NLP model genuinely add value to our credit analysis?'")
            fig_meta_weights = plot_meta_learner_weights(st.session_state.meta_learner_coefs)
            st.pyplot(fig_meta_weights)
    else:
        st.info("Please train the base models first on the 'Base Model Training' page.")

# Page 4: Performance & Stability Analysis
elif st.session_state.page == "Performance & Stability Analysis":
    st.title("4. Evaluate Performance & Stability: Beyond Accuracy")
    st.markdown(f"For financial models, **robustness and stability** are often as critical as raw predictive accuracy. While individual models might perform well in certain periods, ensembles deliver more consistent performance across varying economic regimes and data samples, a key requirement for **SR 11-7** compliance.")

    st.header("a. Six-Way AUC Comparison (V1)")
    st.markdown(f"This chart summarizes the Area Under the Receiver Operating Characteristic Curve (AUC) for all three individual models and three ensemble strategies. AUC provides a single metric for a classifier's ability to distinguish between default and non-default cases across all possible thresholds.")
    st.markdown(r"$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(x)) dx$$")
    st.markdown(r"where TPR is the True Positive Rate and FPR is the False Positive Rate, measuring the classifier's performance across different thresholds.")

    st.header("b. ROC Curves Overlay (V2)")
    st.markdown(f"Visualizing all ROC curves on a single plot allows you to see how ensemble models generally dominate individual models, providing better sensitivity (True Positive Rate) for a given specificity (1 - False Positive Rate) across a range of thresholds. This visually demonstrates the consistent lift provided by ensembling.")

    st.header("c. Bootstrap Confidence Intervals (V5)")
    st.markdown(f"We assess model stability by generating 1,000 bootstrap samples and re-calculating AUC for each. Narrower confidence intervals (CIs) for ensemble models indicate greater robustness and less variance in performance, a crucial quality for models deployed in production credit systems.")
    st.markdown(r"$$\text{CI} = [\text{AUC}_{2.5\%}, \text{AUC}_{97.5\%}]$$")
    st.markdown(r"where $\text{AUC}_{x\%}$ is the $x$-th percentile of AUC scores across bootstrap samples. A smaller $\text{width} = \text{AUC}_{97.5\%} - \text{AUC}_{2.5\%}$ indicates higher stability.")

    if st.session_state.ensembles_trained:
        if st.button("Evaluate All Models", help="Calculates AUCs, Lift, and Bootstrap CIs for all models."):
            with st.spinner("Evaluating models and running stability analysis..."):
                all_probabilities_dict = st.session_state.all_probabilities_dict
                all_auc_scores_dict = {
                    'Fundamental Model': roc_auc_score(st.session_state.y_test, all_probabilities_dict['Fundamental Model']),
                    'Market Signal Model': roc_auc_score(st.session_state.y_test, all_probabilities_dict['Market Signal Model']),
                    'NLP Model': roc_auc_score(st.session_state.y_test, all_probabilities_dict['NLP Model']),
                    'Average Ensemble': roc_auc_score(st.session_state.y_test, all_probabilities_dict['Average Ensemble']),
                    'Voting Ensemble': roc_auc_score(st.session_state.y_test, all_probabilities_dict['Voting Ensemble']),
                    'Stacking Ensemble': roc_auc_score(st.session_state.y_test, all_probabilities_dict['Stacking Ensemble'])
                }
                st.session_state.all_auc_scores_dict = all_auc_scores_dict
                st.session_state.all_probabilities_dict = all_probabilities_dict

                ci_results_df = calculate_bootstrap_cis(st.session_state.y_test, st.session_state.all_probabilities_dict)
                st.session_state.ci_results_df = ci_results_df

                results_df_summary = prepare_model_summary_df(st.session_state.all_auc_scores_dict, ci_results_df)
                st.session_state.results_df_summary = results_df_summary
            st.success("Evaluation complete!")

        if st.session_state.results_df_summary is not None:
            st.markdown(f"### Model Performance Summary")
            st.dataframe(st.session_state.results_df_summary.style.format({
                'AUC': "{:.4f}",
                'AUC Lift': "{:+.4f}",
                '95% CI Lower': "{:.4f}",
                '95% CI Upper': "{:.4f}",
                'CI Width': "{:.4f}"
            }))

            st.markdown(f"### Six-Way AUC Bar Chart (V1)")
            fig_auc_bar = plot_auc_bar_chart(st.session_state.results_df_summary)
            st.pyplot(fig_auc_bar)

            st.markdown(f"### ROC Curves Overlay (V2)")
            fig_roc = plot_roc_curves_overlay(st.session_state.y_test, st.session_state.all_probabilities_dict)
            st.pyplot(fig_roc)
            st.markdown(f"**Financial Interpretation:** The ensemble ROC curves visibly "
                        f"lie above individual model curves, demonstrating their superior "
                        f"ability to discriminate between defaulting and non-defaulting bonds "
                        f"across various thresholds. This consistent dominance is a key "
                        f"indicator of a robust and reliable model for risk management.")

            st.markdown(f"### Bootstrap Confidence Intervals (V5)")
            st.markdown(f"This visualization illustrates the distribution of AUC scores across 1,000 bootstrap samples for each model. The narrower spread (smaller violin plot or box) for ensemble models demonstrates their enhanced stability compared to individual models.")
            fig_ci_violin = plot_bootstrap_violin_plot(st.session_state.ci_results_df)
            st.pyplot(fig_ci_violin)
            st.markdown(f"**Practitioner Warning:** While ensemble lift in AUC might appear modest, their true value lies in **robustness** and **stability**. Ensembles perform more consistently across different market conditions, a critical feature for production credit risk systems.")
    else:
        st.info("Please train the ensemble models first on the 'Ensemble Strategies' page.")

# Page 5: Model Agreement & Triage
elif st.session_state.page == "Model Agreement & Triage":
    st.title("5. Model Agreement & Disagreement: A Triage System for Human Review")
    st.markdown(f"For a Credit Analyst, understanding *when* models agree and *when* they disagree is crucial for operational efficiency and regulatory compliance. This analysis allows you to implement a **triage system** for human review, focusing expert attention on high-uncertainty cases. This directly supports the **SR 11-7** challenger model framework by identifying situations where different 'challenger' views diverge.")

    st.header("a. Defining Default Prediction Threshold")
    st.markdown(f"We'll use a probability threshold to convert continuous default probabilities into binary predictions (default/no-default). This threshold is typically lower in credit risk due to the high cost of missing a default.")

    if st.session_state.ensembles_trained:
        st.session_state.agreement_threshold = st.slider(
            "Select Default Probability Threshold:",
            min_value=0.01, max_value=0.50, value=st.session_state.agreement_threshold, step=0.01,
            help="Adjust the probability threshold for binary default predictions. Lower values are more conservative."
        )

        if st.button("Analyze Model Agreement", help="Analyzes how base models agree or disagree on default predictions."):
            with st.spinner("Analyzing model agreement..."):
                confidence_df, disagreement_cases_df = perform_agreement_analysis(
                    st.session_state.y_test, st.session_state.prob_fund, st.session_state.prob_mkt,
                    st.session_state.prob_nlp, st.session_state.prob_stack,
                    st.session_state.agreement_threshold, st.session_state.X_test,
                    st.session_state.fundamental_features, st.session_state.market_features, st.session_state.nlp_features
                )

                st.session_state.confidence_df = confidence_df
                st.session_state.disagreement_cases_df = disagreement_cases_df
            st.success("Agreement analysis complete!")

        if st.session_state.confidence_df is not None:
            st.markdown(f"### Model Agreement Distribution (V4)")
            st.markdown(f"This stacked bar chart shows the distribution of cases based on how many base models (0, 1, 2, or 3) predict default, along with the actual default rate for each level of agreement. This forms the basis of a 'Green-Yellow-Red' triage system.")
            fig_agreement_bar = plot_model_agreement_stacked_bar(st.session_state.confidence_df)
            st.pyplot(fig_agreement_bar)
            st.markdown(f"**Triage System Interpretation:**")
            st.markdown(f"-   **Green (0/3 models flag default):** High confidence in 'no default'. Minimal human review needed. ")
            st.markdown(f"-   **Yellow (1/3 or 2/3 models flag default):** Cases of disagreement. These require careful human review. The specific models disagreeing indicate where an analyst should look (e.g., if only NLP flags risk, scrutinize management communications). ")
            st.markdown(f"-   **Red (3/3 models flag default):** High confidence in 'default'. Take protective action (e.g., tighten covenants, increase reserves). ")

            st.markdown(f"### Disagreement Case Examples (V7)")
            st.markdown(f"Examining individual cases where models disagree provides actionable insights. This table shows specific examples where base models diverged in their predictions, alongside the true outcome and the features relevant to each model.")
            if not st.session_state.disagreement_cases_df.empty:
                st.dataframe(st.session_state.disagreement_cases_df)
                st.markdown(f"**Financial Interpretation:** These are the most valuable cases for a credit analyst. If the market model flags default but fundamentals are stable, investigate what the market knows. If only the NLP model flags default due to deteriorating tone, scrutinize management communications. This operationalizes the ensemble for proactive risk management.")
            else:
                st.info("No disagreement cases found with the current threshold. Try adjusting the threshold.")
    else:
        st.info("Please train the ensemble models first on the 'Ensemble Strategies' page.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
