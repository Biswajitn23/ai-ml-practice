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

---

## 📅 Day 12 – ROC Curve & AUC
- Learned limitations of accuracy  
- Used probability-based prediction  
- Implemented ROC curve  
- Calculated AUC score  
- Evaluated model discrimination ability  

**File:** `roc_auc_demo.py`

---

## 📅 Day 13 – Precision, Recall & F1 Score
- Learned confusion matrix components  
- Implemented precision and recall  
- Calculated F1 score  
- Understood trade-offs between precision and recall  

**File:** `precision_recall_demo.py`

---

## 📅 Day 14 – Save & Load ML Model
- Trained Random Forest model  
- Saved trained model using joblib  
- Loaded saved model for prediction  
- Learned basics of ML deployment workflow  

**Files:**  
- `save_model_demo.py`  
- `load_model_demo.py`

---

## 📅 Day 15 – ML API with Flask
- Built Flask API for ML model  
- Loaded saved Random Forest model  
- Created prediction endpoint  
- Tested prediction using JSON input  

**File:** `app.py`

---

## 📅 Day 16 – Deploy ML API Online
- Prepared Flask API for deployment  
- Added requirements and Procfile  
- Deployed ML model API on Render  
- Tested prediction using online endpoint  

**Deployment:** Render

---

## 📅 Day 17 – Add Web UI
- Created HTML interface  
- Connected frontend form with ML backend  

---

## 📅 Day 18 – Improved UI & Labels
- Styled UI  
- Converted prediction output to readable medical labels  

---

## 📅 Day 19 – Structured Input Fields
- Replaced textarea with 30 numeric input fields  
- Improved validation  
- Created full-stack ML web app  

---

## 📅 Day 20 – Portfolio-Level Documentation
- Improved project documentation
- Added professional README structure
- Included screenshots and architecture description
- Organized project for better readability and presentation

---

## 📅 Day 21 – Professional README Enhancements
- Added project badges (Python, Flask, ML, Deployment)
- Improved project description
- Added live demo and feature highlights
- Structured documentation for portfolio presentation

---

## 📅 Day 22 – Prediction Confidence Score
- Added probability prediction using `predict_proba`
- Returned confidence score in API response
- Displayed prediction confidence in UI
- Improved model inference output

---

## 📅 Day 23 – API Documentation with Swagger

- Integrated Swagger UI using **Flasgger**
- Generated interactive API documentation
- Enabled testing API endpoints directly from the browser
- Improved developer experience for API usage
### Endpoint
GET `/api-docs`

---

## 📅 Day 24 – Feature Importance API

- Extracted feature importance from the trained **Random Forest model**
- Added API endpoint to expose feature importance
- Enabled model interpretability for understanding important features
- Returned sorted importance values in JSON format
### Endpoint
GET `/feature-importance`

---

## 📅 Day 25 – Feature Importance Visualization

- Added feature importance visualization using **Chart.js**
- Integrated frontend chart with backend `/feature-importance` API
- Displayed model interpretability directly in the web interface
- Allowed users to visually understand which features influence predictions

---

## 📅 Day 26 – Prediction Confidence Visualization

- Added prediction confidence score to the UI
- Implemented a **confidence progress bar**
- Added **color-coded prediction results**