# QuLab: Lab 24: Ensemble Model for Risk Prediction

## 💡 Ensemble Model for Robust Corporate Bond Default Prediction

This Streamlit application, part of the QuantUniversity Lab series (QuLab: Lab 24), demonstrates the power of ensemble modeling for robust corporate bond default prediction. Designed for CFA Charterholders and Credit Analysts, the application guides users through building a sophisticated risk prediction system by combining diverse models and data sources, a practice central to advanced risk management and compliance with regulatory frameworks like **SR 11-7**.

## ✨ Features

This application provides a guided workflow to:

1.  **Data Preparation**:
    *   Simulate and prepare a credit default dataset with three distinct feature sets: **Fundamental**, **Market Signal**, and **NLP**.
    *   Visualize raw data overview and feature set breakdowns.

2.  **Base Model Training**:
    *   Train three diverse base models, each specialized for a different feature set:
        *   **Fundamental Model**: Logistic Regression
        *   **Market Signal Model**: XGBoost
        *   **NLP Model**: LightGBM
    *   Evaluate individual model performance using AUC on the test set.
    *   Visualize **Prediction Correlation Heatmap (V3)** to confirm model diversity (decorrelated errors).

3.  **Ensemble Strategies**:
    *   Implement and understand three key ensemble methods:
        *   **Simple Probability Averaging**: Equal weighting of base model predictions.
        *   **Voting Classifier (Soft Voting)**: Weighted averaging of base model predictions.
        *   **Stacking (Meta-Learner)**: A sophisticated approach where base model predictions serve as features for a meta-learner (Logistic Regression).
    *   Evaluate ensemble model performance using AUC.
    *   Visualize **Meta-Learner Weights (V6)** for the Stacking ensemble, showing the importance assigned to each base model.

4.  **Performance & Stability Analysis**:
    *   Conduct a comprehensive evaluation of all base and ensemble models:
        *   **Six-Way AUC Comparison (V1)**: Bar chart showing AUC scores for all models.
        *   **ROC Curves Overlay (V2)**: Visual comparison of classifier performance across thresholds.
        *   **Bootstrap Confidence Intervals (V5)**: Assess model stability and robustness by displaying 95% confidence intervals and violin plots of AUC scores across 1,000 bootstrap samples.

5.  **Model Agreement & Triage**:
    *   Define a user-adjustable default probability threshold.
    *   Analyze model agreement:
        *   **Model Agreement Distribution (V4)**: Stacked bar chart showing the distribution of cases based on how many base models predict default (0, 1, 2, or 3), forming a "Green-Yellow-Red" triage system for human review.
        *   **Disagreement Case Examples (V7)**: Display specific instances where base models diverge, providing actionable insights for credit analysts.

## 🚀 Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository** (if hosted on GitHub/GitLab):
    ```bash
    git clone https://github.com/your-username/quslab-lab24-ensemble-risk-prediction.git
    cd quslab-lab24-ensemble-risk-prediction
    ```
    *(If not a repository, simply create a directory and place `app.py`, `source.py`, and `requirements.txt` inside it.)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required packages**:
    Create a `requirements.txt` file in the project root with the following content:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    xgboost
    lightgbm
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

1.  **Ensure you are in the project root directory** and your virtual environment is activated.
2.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

3.  **Navigate the application**:
    *   Use the sidebar to navigate between the different lab sections:
        *   `Introduction & Data Preparation`
        *   `Base Model Training`
        *   `Ensemble Strategies`
        *   `Performance & Stability Analysis`
        *   `Model Agreement & Triage`
    *   Follow the instructions on each page, clicking buttons like "Load and Prepare Data", "Train Base Models", etc., to progress through the lab.

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains all data preparation, model training,
|                           # evaluation, and plotting utility functions.
└── requirements.txt        # List of Python dependencies
```

*   **`app.py`**: This is the core Streamlit application. It manages the UI, session state, and calls functions from `source.py` to perform computations and visualizations.
*   **`source.py`**: This file encapsulates all the backend logic, including:
    *   `prepare_credit_data()`: Simulates and preprocesses the dataset.
    *   `train_individual_models()`: Trains the Logistic Regression, XGBoost, and LightGBM base models.
    *   `train_ensemble_strategies()`: Implements and trains the Averaging, Voting, and Stacking ensembles.
    *   `calculate_bootstrap_cis()`: Performs bootstrap resampling for stability analysis.
    *   `prepare_model_summary_df()`: Aggregates model performance metrics.
    *   `perform_agreement_analysis()`: Conducts the model agreement and disagreement analysis.
    *   Various `plot_*` functions: Generate the `matplotlib`/`seaborn` visualizations displayed in the app.

## 💻 Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/)
*   **Language**: Python 3.8+
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**:
    *   [Scikit-learn](https://scikit-learn.org/): For various utilities, preprocessing, Logistic Regression, Voting and Stacking classifiers.
    *   [XGBoost](https://xgboost.readthedocs.io/): Gradient Boosting model for market signals.
    *   [LightGBM](https://lightgbm.readthedocs.io/): Gradient Boosting model for NLP features.
*   **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure tests pass (if any).
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You'll need to create a `LICENSE` file in your project root if you choose to include one.)*

## 📧 Contact

For any questions or feedback, please reach out to:

*   **QuantUniversity** - [info@quantuniversity.com](mailto:info@quantuniversity.com)
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)


## License

## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
