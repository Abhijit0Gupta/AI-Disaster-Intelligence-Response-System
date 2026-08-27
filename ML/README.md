# Machine Learning Module

## Purpose

This module is responsible for analyzing disaster-related data and predicting the severity of a flood disaster.

The ML model will be part of the core intelligence system of the AI Disaster Intelligence & Response System.

---

## Objective

The model will receive disaster-related numerical information and predict the disaster severity.

Possible severity levels:

- Low
- Medium
- High
- Critical

---

## Planned Input Features

The initial model may use the following features:

- rainfall_mm
- water_level_m
- affected_population
- damage_percentage

The final features will depend on the selected dataset.

---

## Target Variable

The model will predict:

severity

---

## ML Pipeline

Cleaned Dataset
        ↓
Feature Selection
        ↓
Target Selection
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Saving
        ↓
Prediction

---

## Planned Files

- model_training.ipynb
- predict.py
- disaster_model.pkl

---

## Output

The model should return one of the following severity levels:

- low
- medium
- high
- critical

The output will later be used by the Priority Engine and Resource Allocation Engine.

---

## Integration

The backend will eventually call the ML prediction module.

Expected flow:

Frontend
    ↓
Backend
    ↓
ML Prediction Module
    ↓
Severity Result
    ↓
Priority Engine
    ↓
Resource Allocation Engine
    ↓
Backend Response
    ↓
Frontend