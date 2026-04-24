# Telecom-Churn-Predictor
End-to-end machine learning project to predict customer churn 30 days in advanced - built with XGBoost and scikit-learn

## 📑 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Project Structuire](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Monitoring](#monitoring)

## Overview
A telecom company loses ~27% of its customers annually. This project builds a **binary classification model** that identifies at-risk customers before they churn, enabling retention teams to intervene proactively.

| | |
|---|---|
| **Task** | Binary Classification (churn / no churn) |
| **Prediction horizon** | 30 days before churn event |
| **Primary metric** | AUC-ROC |
| **Dataset** | 7,043 customers × 21 features |
| **Class balance** | 73.5% no churn / 26.5% churn |

## Results
| Model | CV AUC-ROC | Std |
| --- | --- | --- |
| Logistic Regression (baseline) | 0.849 | ± 0.013 |
| Random Forest | 0.847 | ± 0.014 |
| Gradient Boosting | 0.844 | ± 0.012 |
| **XGBoost (selected)** | **0.8503** | **± 0.01** |
| LightGBM | 0.831 | ± 0.015 |

**Holdout test set (XGBoost, threshold=0.4):**
| Metric | Score | Target |
| --- | ---| --- |
| AUC-ROC | 0.854 | ≥ 0.85 ✅ |
| Recall | 0.811 | ≥ 0.80  ✅ |
| Precision | 0.52 | — |
| F1-Score | 0.63 | — |

## Project Structure
```bash
project-name/
Telecom-Churn-Predictor
```
