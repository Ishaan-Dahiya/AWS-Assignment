# AWS-Assignment
# Leveraging Machine Learning and Predictive Analytics in Healthcare: Classification Modeling for Liver Cirrhosis Staging

## 1. Project Overview
This project explores the use of machine learning models to predict liver cirrhosis stages using clinical and biochemical data. We used multiple machine learning algorithms, including Decision Trees, Random Forest, SVM, Bagging, and Gradient Boosting, to enhance early diagnosis and treatment.

## 2. Objective
The primary objective is to accurately predict liver cirrhosis stages using clinical data. This can aid healthcare professionals in non-invasive diagnosis and treatment planning, ultimately improving patient outcomes.

## 3. Dataset
The dataset was obtained from [Kaggle Liver Cirrhosis Stage Classification](https://www.kaggle.com/datasets/aadarshvelu/liver-cirrhosis-stage-classification/data) and includes features like:
- Bilirubin
- Albumin
- Prothrombin time
- Platelet count

## 4. Methodology

### 4.1 Data Collection & Preprocessing
- Handled missing values using median imputation.
- Applied normalization for features with high variance.
- Encoded categorical variables like `sex` and `ascites`.

### 4.2 Exploratory Data Analysis
We conducted an in-depth exploratory analysis to understand the relationships between the variables.
- **Heatmap of Correlations**: 
  ![Heatmap of Correlations](images/heatmap.png)

- **Age Distribution**: 
  ![Age Distribution](images/age_distribution.png)

### 4.3 Model Building
- Implemented multiple models: Decision Trees, Random Forest, SVM, Bagging, and Gradient Boosting.
- Hyperparameter tuning was performed using `GridSearchCV` and `RandomizedSearchCV` to optimize each model's performance.

### 4.4 Model Evaluation
The models were evaluated based on Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE). The following table summarizes the performance of the models:

| Model              | MAPE  | RMSE  |
|--------------------|-------|-------|
| Random Forest      | 0.042 | 0.291 |
| Bagging            | 0.048 | 0.314 |
| Decision Tree      | 0.069 | 0.393 |
| Gradient Boosting  | 0.128 | 0.507 |
| SVM (RBF)          | 0.431 | 1.040 |

## 5. Tools and Technologies
- **Python**: Used for data manipulation, analysis, and modeling.
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Jupyter Notebook**: For interactive coding and data exploration.

## 6. Results and Key Insights
- Random Forest and Bagging models outperformed the others, achieving the lowest error rates.
- Feature importance analysis highlighted **prothrombin time** and **age** as key predictors.
- Below is a **Feature Importance Plot** for the Random Forest model:
  
  ![Feature Importance](images/feature_importance.png)

## 7. Deliverables
- **Code**: Available in the `notebooks` folder.
- **Report**: [Liver Cirrhosis Analysis Report](report.pdf)
- **Presentation**: A summary of key findings in the `presentation` folder.

## 8. Conclusion
The project demonstrates how machine learning can be used to predict liver cirrhosis stages accurately. By utilizing Random Forest and Bagging models, we achieved high accuracy, making this methodology applicable in clinical diagnostics.

## 9. How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/liver-cirrhosis-staging.git
