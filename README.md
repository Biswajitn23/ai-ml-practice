# AI / ML Practical Learning Journey 🚀

This repository documents my step-by-step practical learning of Artificial Intelligence and Machine Learning.

---

## 📅 Day 1 – Data Loading & Visualization
- Loaded CSV data using Pandas  
- Explored dataset structure  
- Created a new feature (Average marks)  
- Visualized student performance using a bar chart  

**File:** `student_analysis.py`

---

## 📅 Day 2 – Exploratory Data Analysis (EDA)
- Inspected dataset using `info()` and `describe()`  
- Detected missing values  
- Handled missing data using mean imputation  
- Visualized score distributions using box plots  

**File:** `eda_analysis.py`

---

## 📅 Day 3 – First Machine Learning Model (Regression)
- Separated features and target  
- Applied train-test split  
- Trained a Linear Regression model  
- Evaluated using Mean Absolute Error (MAE)  
- Interpreted model coefficients  

**File:** `ml_model.py`

---

## 📅 Day 4 – Classification with Logistic Regression
- Converted numerical averages into Pass/Fail labels  
- Trained a Logistic Regression classifier  
- Evaluated model using Accuracy  
- Analyzed predictions using Confusion Matrix  
- Checked class distribution before training  

**File:** `classification_model.py`

---

## 📅 Day 5 – KNN with Feature Scaling
- Learned why scaling is important for distance-based models  
- Standardized features using `StandardScaler`  
- Trained K-Nearest Neighbors (KNN) classifier  
- Evaluated model accuracy and confusion matrix  

**File:** `knn_model.py`

---

## 📅 Day 6 – Model Comparison & Hyperparameter Tuning
- Compared Logistic Regression and KNN models  
- Tested multiple K values in KNN  
- Selected best K based on accuracy  
- Learned basics of hyperparameter tuning  

**File:** `model_comparison.py`

---

## 📅 Day 7 – Cross-Validation
- Learned why a single train-test split can be unreliable  
- Used cross-validation for more stable model evaluation  
- Evaluated Logistic Regression and KNN using multiple folds  
- Understood limitations of cross-validation on small datasets  
- Adjusted K in KNN to match training fold size  

**File:** `cross_validation.py`

---

## 📅 Day 8 – Overfitting vs Underfitting
- Learned difference between overfitting and underfitting  
- Compared training vs testing accuracy  
- Observed effect of different K values in KNN  
- Understood model complexity trade-offs  

**File:** `overfitting_demo.py`

---

## 📅 Day 9 – Bias vs Variance
- Learned concept of bias and variance  
- Understood relationship with model complexity  
- Demonstrated bias-variance tradeoff using KNN  
- Visualized training and testing error curves  

**File:** `bias_variance_demo.py`

---

## 📅 Day 10 – Decision Trees
- Learned tree-based classification  
- Understood feature-based splitting  
- Visualized decision tree structure  
- Observed impact of tree depth on overfitting  

**File:** `decision_tree_demo.py`

---

## 📅 Day 11 – Random Forest
- Learned ensemble learning concept  
- Implemented Random Forest classifier  
- Compared stability with Decision Tree  
- Visualized feature importance  
- Understood variance reduction through averaging  

**File:** `random_forest_demo.py`
