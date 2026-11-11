# Disaster Type Prediction & Interactive Model Auditor (Explainable AI)

A Senior Project that integrates **Multi-class Disaster Event Classification (Random Forest, LightGBM, and XGBoost pipelines)** with an **Agentic Model Auditor** and **Explainable AI (SHAP)**.

---

## 🌟 Key Features

1. **High Performance ML Pipeline**:
   - Random Forest, LightGBM, and XGBoost classifiers, each in a **SMOTETomek** class-balancing pipeline.
   - Random Forest: **91.70% Test Accuracy** and **0.8518 F1-Macro** across 8 disaster categories; the leaderboard tab shows all three models with bootstrap 95% CIs and McNemar significance tests.
2. **Interactive Ground Truth Model Auditor**:
   - Challenges the pre-trained model with 3,495 real historical events from the EM-DAT test set.
   - Live side-by-side comparison: **Model Prediction vs. Ground Truth**, or all three models at once in Compare mode.
   - Model Disagreement Radar surfaces the test cases where the three models predict different classes.
   - Real-time Session Scoreboard tracking accuracy and error patterns.
3. **Explainable AI (SHAP TreeExplainer)**:
   - Quantifies the impact of physical features (magnitude scale, event duration, timing, geographic region).
   - Explains *why* the model made each decision, both for correct cases and hard misclassified cases.
4. **LLM Auditor Chat (Claude or Gemini)**:
   - Ask questions about the active case; answers are grounded in its predictions and SHAP drivers.
   - Choose Anthropic Claude or Google Gemini in the sidebar and paste an API key, or save it once in `.streamlit/secrets.toml` (git-ignored) as `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` so it loads on every start. Environment variables and a `.env` file work too; `LLM_PROVIDER = "gemini"` or `"claude"` picks the starting provider.
   - Falls back to a **Local Auditor Engine** summary when no API key is provided.

---

## 📂 Project Structure

```text
disaster-type-prediction-1/
├── app.py                      # Streamlit Interactive Web Application
├── Dockerfile                  # Container definition for reproducible deployment
├── requirements.txt            # Python dependencies
├── .streamlit/config.toml      # Streamlit theme & server settings
├── models/                     # Production artifacts
│   ├── rf_pipeline.joblib      # Trained best Random Forest Pipeline
│   ├── lgb_pipeline.joblib     # LightGBM Pipeline
│   ├── xgb_pipeline.joblib     # XGBoost Pipeline
│   ├── multi_model_bank.joblib # 3,495 test events with precomputed predictions from all 3 models
│   ├── multi_model_metadata.json # Feature names, leaderboard, McNemar results
│   ├── metadata.json           # (legacy, single-model) feature names and metrics
│   └── ground_truth_bank.json  # (legacy, single-model) test events with ground truth
├── src/                        # Modular production code
│   ├── models/                 # Multi-model layer used by the app
│   │   ├── registry.py         # Loads the case bank, runs multi-model audits
│   │   ├── base.py             # Common model interface
│   │   ├── rf_model.py / lgb_model.py / xgb_model.py  # Predict + SHAP per model
│   │   └── utils.py            # Class names & natural-language SHAP explanations
│   ├── agent.py                # LLM chat (Claude / Gemini) + local fallback
│   ├── export_all_models.py    # Trains LightGBM/XGBoost, builds multi-model bank & metadata
│   ├── export_model.py         # (legacy) Random Forest pipeline export
│   └── predictor.py / explainer.py / benchmark.py  # (legacy, single-model; not used by app.py)
├── notebooks/                  # Academic research & experimental record
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_EDA.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_model_export_and_benchmark.ipynb
└── data/                       # Datasets
    ├── cleaned/
    └── processed/
```

---

## 🚀 Quickstart & Usage

### 1. Run Locally (Streamlit)

Activate your Python virtual environment and run (to regenerate the LightGBM/XGBoost pipelines and the multi-model bank, first run `python -m src.export_all_models`; it needs `data/processed/disaster_processed_feature_selected.csv`):
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run with Docker

Build and launch the container with a single command:
```bash
docker build -t disaster-model-auditor .
docker run -p 8501:8501 disaster-model-auditor
```

---

## 🎯 Target Disaster Classes (8 Classes)
1. Drought
2. Earthquake
3. Epidemic
4. Extreme Temperature
5. Flood
6. Landslide
7. Storm
8. Wildfire
