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

---

## 📅 Day 27 – Prediction History Dashboard

- Implemented prediction history logging
- Displayed previous predictions in the UI
- Added timestamp tracking
- Created a simple monitoring dashboard

---

## 📅 Day 28 – Top-10 Feature Importance Chart

- Improved feature importance visualization
- Displayed only the top 10 most important features
- Converted chart to horizontal bar chart for better readability

---

## 📅 Day 29 – System Monitoring Endpoint

- Added `/health` endpoint for monitoring application status
- Returned system status, model name, and prediction count
- Enables quick health checks for the deployed ML service

---

## 📅 Day 30 – Model Information Endpoint

- Added `/model-info` endpoint
- Returns details about the deployed machine learning model
- Helps developers understand model configuration

Example response:

{
  "model_name": "Breast Cancer Classifier",
  "algorithm": "RandomForestClassifier",
  "library": "Scikit-Learn",
  "input_features": 30,
  "problem_type": "Binary Classification"
}

---

## 📅 Day 31 – Feature Importance Analysis

- Extracted feature importance values from Random Forest model
- Ranked features based on their contribution to predictions
- Displayed the top important features for model interpretation

File: `feature_importance_demo.py`

---

## 📅 Day 32 – PCA (Dimensionality Reduction)

- Applied Principal Component Analysis (PCA) for dimensionality reduction
- Reduced high-dimensional data into 2 principal components
- Visualized data in 2D space for better understanding

File: `pca_demo.py`

---

## 📅 Day 33 – K-Means Clustering

- Implemented K-Means clustering for unsupervised learning
- Grouped data points into clusters without labels
- Visualized clusters and centroid positions

File: `kmeans_demo.py`

---

## 📅 Day 34 – Elbow Method for Optimal Clusters

- Implemented elbow method to determine optimal number of clusters  
- Plotted inertia vs number of clusters  
- Identified best K value visually  

File: `elbow_method_demo.py`

---

## 📅 Day 35 – Gradient Boosting

- Implemented Gradient Boosting classifier  
- Learned boosting concept using weak learners  
- Evaluated model performance on dataset  

File: `gradient_boosting_demo.py`

---

## 📅 Day 36 – Hyperparameter Tuning with GridSearchCV

- Used GridSearchCV for automatic hyperparameter tuning  
- Tested multiple parameter combinations  
- Selected best model based on performance  

File: `grid_search_demo.py`

---

## 📅 Day 37 – Pipeline + Hyperparameter Tuning

- Combined preprocessing and model using Pipeline  
- Integrated GridSearchCV with pipeline  
- Ensured consistent data preprocessing during tuning  

File: `pipeline_gridsearch_demo.py`

---

## 📅 Day 38 – Advanced Model Comparison

- Compared multiple ML models in a single script  
- Evaluated Logistic Regression, Random Forest, and Gradient Boosting  
- Displayed accuracy for each model  

File: `advanced_model_comparison.py`

---

## 📅 Day 39 – Confusion Matrix Visualization

- Visualized confusion matrix using heatmap  
- Improved interpretability of classification results  
- Used seaborn for better visualization  

File: `confusion_matrix_visualization.py`

---

## 📅 Day 40 – Learning Curve Analysis

- Generated learning curves to analyze model performance  
- Compared training and validation scores  
- Identified overfitting and underfitting behavior  

File: `learning_curve_demo.py`

---

## 📅 Day 41 – Project Setup (Student Performance Prediction)

- Started end-to-end ML project  
- Selected problem: Student Performance Prediction  
- Loaded dataset and explored basic structure  
- Prepared project folder structure  

File: `project_setup.py`

---

## 📅 Day 42 – Exploratory Data Analysis (EDA)

- Performed EDA on student dataset  
- Checked missing values and data types  
- Visualized feature distributions  
- Analyzed data patterns  

File: `project_eda.py`

---

## 📅 Day 43 – Feature Engineering

- Created new feature: Average marks  
- Handled missing values using mean imputation  
- Prepared dataset for model training  

File: `feature_engineering.py`

---

## 📅 Day 44 – Model Training

- Converted average marks into classification labels  
- Split dataset into training and testing sets  
- Trained Logistic Regression model  
- Generated predictions  

File: `model_training.py`

---

## 📅 Day 45 – Model Evaluation

- Evaluated model using accuracy score  
- Generated confusion matrix  
- Calculated precision, recall, and F1 score  
- Analyzed classification performance  

File: `model_evaluation.py`

---

## 📅 Day 46 – Random Forest for Project

- Implemented Random Forest classifier for project  
- Compared performance with Logistic Regression  
- Improved prediction stability  
- Evaluated model accuracy  

File: `random_forest_project.py`

---

## 📅 Day 47 – Hyperparameter Tuning for Project

- Applied GridSearchCV for Random Forest tuning  
- Tested different hyperparameter combinations  
- Selected best-performing configuration  
- Improved model performance  

File: `project_hyperparameter_tuning.py`

---

## 📅 Day 48 – Model Saving for Project

- Saved trained Random Forest model using joblib  
- Stored trained model for future predictions  
- Learned persistence of ML models  

File: `save_project_model.py`

---

## 📅 Day 49 – Load Model & Predict

- Loaded saved Random Forest model using joblib  
- Used saved model for predictions  
- Simulated inference workflow for deployment  

File: `load_project_model.py`

---

## 📅 Day 50 – Flask API for Student Prediction

- Built Flask API for student performance prediction  
- Loaded trained Random Forest model  
- Created `/predict` endpoint for inference  
- Returned predictions using JSON responses  

File: `student_api.py`

---

## 📅 Day 51 – Introduction to Neural Networks

- Built first Artificial Neural Network (ANN) using TensorFlow/Keras  
- Learned basics of input, hidden, and output layers  
- Trained neural network on breast cancer dataset  
- Evaluated model accuracy  

File: `neural_network_demo.py`

---

## 📅 Day 52 – Training Visualization for Neural Network

- Visualized neural network training performance  
- Plotted training and validation accuracy curves  
- Observed learning behavior across epochs  
- Learned basics of monitoring deep learning models  

File: `nn_training_visualization.py`

---

## 📅 Day 53 – Dropout Regularization

- Added Dropout layers to reduce overfitting  
- Improved neural network generalization  
- Learned regularization techniques in deep learning  
- Compared training stability with dropout  

File: `dropout_regularization_demo.py`

---

## 📅 Day 54 – Early Stopping in Neural Networks

- Implemented EarlyStopping callback in Keras  
- Prevented unnecessary training epochs  
- Reduced overfitting during training  
- Learned automatic training stopping techniques  

File: `early_stopping_demo.py`

---

## 📅 Day 55 – Model Saving & Loading for Neural Networks

- Saved trained neural network model using Keras  
- Loaded saved model for future predictions  
- Learned persistence workflow for deep learning models  
- Simulated deployment-ready inference process  

File: `nn_model_save_load.py`

---

## 📅 Day 56 – Image Classification with MNIST

- Introduced image classification using neural networks  
- Loaded MNIST handwritten digit dataset  
- Trained a neural network to recognize digits (0–9)  
- Evaluated model accuracy on unseen images  

File: `mnist_classification.py`