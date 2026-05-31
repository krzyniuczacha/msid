# Diabetes Risk Prediction: Fuzzy Logic vs Machine Learning

## Project Overview
This project was developed as part of the **Systems and Decision-Making Methods** university course. The primary objective is to design, implement, and evaluate two fundamentally different approaches to binary classification for predicting diabetes risk in patients:
1.  **Fuzzy Logic Expert Systems** built using the `scikit-fuzzy` library.
2.  **Classical Machine Learning Models** implemented via `scikit-learn`.

The project explores multiple configurations of Fuzzy Inference Systems (FIS) by testing the impact of input attribute count, membership function (MF) granularity, and MF shapes against standard ML baselines.

## Dataset
The system utilizes the widely known `diabetes.csv` dataset (Pima Indians Diabetes Database) containing 768 medical cases.

**Data Preprocessing & Engineering:**
* **Missing Value Imputation:** Biological anomalies and missing entries recorded as `0` (e.g., Glucose, Blood Pressure, BMI, Insulin, Skin Thickness) were replaced with the column-wise median values.
* **Fuzzy System Constraints:** Input values for BMI and Age were safely bounded using `clip()` to ensure robust fuzzy universe evaluation.
* **ML Scaling:** Features were normalized using `StandardScaler` to optimize model training, particularly for Logistic Regression.
* **Data Split:** The dataset was divided into an 80% training set and a 20% testing set to ensure unbiased evaluation.

## Experimental Setup

### 1. Fuzzy Logic Module
Three distinct experiments were conducted to evaluate the FIS performance:
* **Experiment 1 (Attribute Count):** Comparing systems using 1 attribute (Glucose), 2 attributes (Glucose + BMI), and 3 attributes (Glucose + BMI + Age).
* **Experiment 2 (MF Granularity):** Evaluating the impact of linguistic terms partition: Binary (2 MFs), Baseline (3 MFs), and Highly Granular (4 MFs).
* **Experiment 3 (MF Shape Type):** Comparing triangular (`trimf`), trapezoidal (`trapmf`), and Gaussian (`gaussmf`) membership functions.

### 2. Machine Learning Module (Baseline)
Three reference models were trained on the identical dataset partition:
* **Logistic Regression** (with `max_iter=1000`)
* **Decision Tree Classifier** (tuned with `max_depth=5`)
* **Random Forest Classifier** (ensemble of 100 estimators)

## Evaluation and Results
Each configuration is evaluated using standard classification metrics: **Accuracy, Precision, Recall, and F1-score**, along with the **inference execution time (ms)**.

The main execution script outputs a structured terminal summary table and automatically exports comprehensive evaluation charts:
* `fuzzy_experiments.png`: Bar chart comparison of accuracy, F1-score, and computational times.
* `confusion_matrices.png`: A comprehensive grid layout displaying confusion matrices for all 11 evaluated fuzzy configurations and ML models.

## Technologies Used
* **Language:** Python
* **Data Analysis & ML:** `pandas`, `numpy`, `scikit-learn`
* **Fuzzy Logic:** `scikit-fuzzy`
* **Data Visualization:** `matplotlib`, `seaborn`
