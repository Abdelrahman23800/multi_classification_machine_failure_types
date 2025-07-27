
# Predicting Machine Failure Types Using Machine Learning

This project focuses on building a machine learning model to classify **types of machine failures** based on various operational features. The model is trained and evaluated using supervised learning techniques and handles class imbalance with **SMOTE (Synthetic Minority Oversampling Technique)**.  

## 📊 Project Overview

- **Goal:** Predict the type of machine failure (e.g., Heat Dissipation Failure, Power Failure, Tool Wear Failure, Overstrain Failure, Random Failures) to reduce downtime.
- **Dataset:** Simulated sensor data representing industrial machine conditions and corresponding failure types.
- **Modeling Approach:** Multi-class classification using different ML algorithms.
- **Challenge Solved:** 
  - **Multi-label to multi-class conversion:** Converted one-hot encoded labels into single-label categorical output.
  - **Class Imbalance:** Solved using SMOTE oversampling technique.

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)

## 🔍 Features Used

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

## 📈 Models Tested

- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Logistic Regression
- decision tree

**Best Accuracy Achieved: 99.2% using Random Forest**

## ⚙️ Workflow

1. **Exploratory Data Analysis (EDA)**
2. **Data Preprocessing:**
   - Label conversion
   - Feature scaling
   - SMOTE oversampling
3. **Model Training & Evaluation**
4. **Confusion Matrix & Classification Report**
5. **Model Comparison**

## 📂 Project Structure

```
📁 Machine-learning/
├── Version_2.ipynb        # Main Jupyter Notebook
├── README.md              # Project overview and documentation
```

## 📌 Key Takeaways

- SMOTE can significantly improve model performance for imbalanced multi-class problems.
- Ensemble models like Random Forest and XGBoost perform well on industrial sensor data.
- Proper preprocessing and label engineering are critical to multi-class classification success.

## ✅ Future Work

- Deploy model using Streamlit.
- Add real-time failure detection simulation.
- Perform hyperparameter tuning for further improvement.

---

If you found this project helpful or inspiring, feel free to ⭐ star the repo!
