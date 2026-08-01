# Titanic Survival Prediction
 
End-to-end machine learning project for predicting passenger survival in the Kaggle Titanic competition — from raw data to a leak-safe, fully reproducible prediction pipeline.
 
**Competition:** https://www.kaggle.com/competitions/titanic
**Kaggle Notebook:** https://www.kaggle.com/code/neckpuck/titanic-survival-prediction
 
## Key Findings
 
- Survival strongly depended on the interaction between class and sex, not just either factor alone: third-class women survived at a noticeably lower rate than the "women and children first" narrative alone would suggest.
- Permutation importance (top 3 features) for Random Forest: **Sex, Prefix, Pclass**, while SVC: **Pclass, Sex, Age**.

## Approach
 
1. **Exploratory Data Analysis** — missing values, distributions, survival rate by feature
2. **Feature Engineering** — Title extraction from `Name`, Deck extraction from `Cabin`, Family Size / Is Alone from `SibSp`/`Parch`, wrapped in a custom `scikit-learn` transformer
3. **Leak-safe Preprocessing Pipeline** — all feature engineering and preprocessing (imputation, scaling, encoding) is fit *inside* cross-validation, never on the full dataset beforehand
4. **Baseline Model Comparison** — six classical models evaluated under identical, leak-safe conditions
5. **Hyperparameter Tuning** — `RandomizedSearchCV` / `GridSearchCV` over the full pipeline (feature engineering + preprocessing + model), so tuning itself never sees held-out data in a preprocessed form
6. **Error Analysis** — per-feature and per-feature-interaction error rates on a held-out test set, plus cross-model agreement analysis
7. **Permutation Importance** — feature importance computed on the held-out test set, at the level of engineered features (Title/Deck/Family Size)
8. **Final Submission** — predictions generated from raw `test.csv`

## Models Compared
 
- Stochastic Gradient Descent Classifier
- Logistic Regression
- Linear Support Vector Machine
- Support Vector Machine (RBF kernel)
- Random Forest
- K-Nearest Neighbors

## Evaluation Metrics
 
- Accuracy, Precision, Recall, F1-score
- Precision-Recall Curve, ROC Curve, ROC AUC
- Confusion Matrix

## Results
 
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| SGD | 0.82 | 0.78 | 0.74 | 0.76 | 0.87 |
| Logistic Regression | 0.82 | 0.78 | 0.75 | 0.76 | 0.87 |
| Linear SVC | 0.82 | 0.78 | 0.75 | 0.77 | 0.87 |
| SVC (RBF) | 0.82 | 0.80 | 0.71 | 0.75 | 0.86 |
| Random Forest | 0.81 | 0.76 | 0.72 | 0.74 | 0.86 |
| KNN | 0.81 | 0.79 | 0.67 | 0.73 | 0.85 |
 
**Public Kaggle Score (tuned Random Forest): 0.77990**
 
## How to Run
 
Download `train.csv` and `test.csv` from the [competition data page](https://www.kaggle.com/competitions/titanic/data) into the project directory, then open `titanic-survival-prediction.ipynb` and run all cells.

## License
 
MIT — see [LICENSE](LICENSE)
 