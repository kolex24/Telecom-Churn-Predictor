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
