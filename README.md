# Credit Default Prediction

This project applies machine learning to predict credit default risk. The main goal is to catch as many defaulters as possible (high recall) while keeping a reasonable precision level. The dataset contains information on past credit card clients, including demographic, payment, and financial details. The target variable is `default payment next month` where 1 → default and 0 → non-default.

The project followed an experimental workflow. Exploratory Data Analysis (EDA) was done to check distributions of categorical and numerical features and clean anomalies (e.g., recoding invalid MARRIAGE values). Feature engineering was applied to create new features such as `PAY_max`, `PAY_mean`, `PAY_late_count` from payment history, `BILL_mean`, `BILL_max`, `BILL_trend` from bill amounts, `PAY_AMT_mean`, `PAY_ratio_mean` from payment amounts, and `BILL_limit_ratio` from bill vs credit limit. StandardScaler was used to standardize numerical features.

The original dataset was highly imbalanced (~78% non-default, 22% default). SMOTE (Synthetic Minority Oversampling Technique) was applied to balance classes. Baseline models including Logistic Regression and Random Forest (with class weighting) were tested before feature engineering. Models were rebuilt after feature engineering to improve performance. Threshold tuning was performed, evaluating thresholds from 0.1 to 0.9 in steps of 0.05, and the threshold maximizing F1-score while boosting recall was selected.

## Results Summary

- **Logistic Regression** (original features, no SMOTE):  
  Confusion Matrix: `[[3263 1410], [502 825]]`  
  Precision (defaulters): 0.37, Recall: 0.62, F1: 0.46, Accuracy: 0.68, ROC-AUC: 0.708  

- **Random Forest** (original features, no SMOTE):  
  Confusion Matrix: `[[4413 260], [873 454]]`  
  Precision (defaulters): 0.64, Recall: 0.34, F1: 0.44, Accuracy: 0.81, ROC-AUC: 0.760  

- **Random Forest with SMOTE** (original features):  
  Confusion Matrix: `[[4158 515], [706 621]]`  
  Precision (defaulters): 0.55, Recall: 0.47, F1: 0.50, Accuracy: 0.80, ROC-AUC: 0.751  

- **Random Forest** (feature engineered, no SMOTE):  
  Confusion Matrix: `[[4394 273], [908 418]]`  
  Precision (defaulters): 0.60, Recall: 0.32, F1: 0.41, Accuracy: 0.80, ROC-AUC: 0.753  

- **Random Forest with feature engineered and SMOTE**:  
  Confusion Matrix: `[[4041 626], [714 612]]`  
  Precision (defaulters): 0.49, Recall: 0.46, F1: 0.48, Accuracy: 0.78, ROC-AUC: 0.744  

- **Random Forest with SMOTE and threshold tuning**:  
  Best threshold: 0.45  
  Confusion Matrix: `[[4009 664], [637 690]]`  
  Precision (defaulters): 0.51, Recall: 0.52, F1: 0.515, Accuracy: 0.78, ROC-AUC: 0.751  

**Interpretation:** Logistic Regression achieves the highest recall (0.62), meaning it catches the most defaulters but with lower precision (more false positives). Random Forest with SMOTE and threshold 0.45 provides the best operational trade-off: recall ≈ 0.52, precision ≈ 0.51, F1 ≈ 0.515, ROC-AUC ≈ 0.751. This configuration balances catching defaulters while reducing false positives compared to logistic regression.

## Repository Structure

credit-risk-model/
│
├── notebooks/
│ ├── experiments.ipynb # raw experiments, multiple models
│ ├── report.ipynb # eplanation
│
├── data/ # (optional) raw/processed data
│
│
├── README.md # project overview
## Next Steps

- Build a pipeline for preprocessing → resampling → training → evaluation  
- Experiment with other models (XGBoost, LightGBM, etc.)  
- Add a deployment-ready script (e.g., using Flask or FastAPI)  

## Author

**Musediku Oluwafisayomi**  
Data Analyst | Aspiring Machine Learning Engineer
