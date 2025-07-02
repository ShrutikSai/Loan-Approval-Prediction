# 🏦 Loan Approval Prediction using Machine Learning

Predicting loan approval status using machine learning models with end-to-end preprocessing, EDA, and model explainability.

---

## 📌 Overview

This project aims to build a machine learning pipeline to predict whether a loan application will be approved or rejected based on applicant details. The project includes data preprocessing, handling imbalanced classes using SMOTE, model comparison, and basic explainability using Decision Trees (SHAP-ready).

---

## 📊 Dataset

- **Source**: [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Rows**: 614 applicants  
- **Target Variable**: `Loan_Status` (Y/N)

---

## 🧰 Tech Stack

- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SMOTE, SHAP (optional)  
- **Tools**: Jupyter Notebook / Google Colab  

---

## 🔍 Exploratory Data Analysis

- Distribution plots (Histogram, Violin)
- Missing value analysis
- Correlation matrix (Heatmap)
- Bivariate analysis (Categorical vs Categorical, Categorical vs Numerical, etc.)

---

## 🛠️ Data Preprocessing

- Dropped unnecessary columns (Loan_ID)
- Filled missing values (mode for categorical, mean for numerical)
- One-hot encoding
- Outlier removal using IQR
- Square root transformation to normalize skewed features
- Applied SMOTE for class imbalance
- MinMaxScaler for normalization

---

## 🤖 Models Trained

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naive Bayes (Categorical & Gaussian)  
- Decision Tree  
- Random Forest  
- Gradient Boosting

---

## 📈 Results

| Model               | Accuracy (%) |
|--------------------|--------------|
| Gradient Boosting  | **80.31**    |
| K-Nearest Neighbors| 76.77        |
| Random Forest      | 73.62        |
| Decision Tree      | 73.23        |
| Logistic Regression| 66.53        |
| Naive Bayes        | 64–66        |
| SVM                | 62.99        |

> ✅ Best Model: Gradient Boosting  
> 🔍 Interpretability: In-built via Decision Tree + SHAP-ready

---

## 🚀 Future Enhancements

- Integrate SHAP for visual model interpretability  
- Deploy model using Flask/Streamlit  
- Improve feature engineering (e.g., feature importance pruning)  
- Try advanced models like XGBoost or LightGBM

---

## ▶️ How to Run

1. Clone the repository  
2. Open the Jupyter Notebook or Colab  
3. Install dependencies (e.g., `pip install -r requirements.txt`)  
4. Run all cells in sequence  

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Credits

- Dataset by [Kaggle Datasets](https://www.kaggle.com/datasets)
- Notebook reference: inspired by Caesar Mario’s work with major extensions and model tuning
