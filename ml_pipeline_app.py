import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# Step 1: Data Upload/Entry
st.title("Machine Learning Pipeline App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
data = None

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())
else:
    st.write("Or, enter data manually:")
    # Optionally, let the user manually enter data if no CSV is uploaded.

# Step 2: Preprocessing Options
st.sidebar.header("Preprocessing Options")
scaling_option = st.sidebar.selectbox("Scaling Method", ["None", "Standard Scaling", "Min-Max Scaling"])

# Step 3: Data Splitting Options
st.sidebar.header("Data Splitting Options")
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
validation_type = st.sidebar.selectbox("Validation Type", ["Holdout", "K-Fold Cross-Validation"])
k_folds = 1
if validation_type == "K-Fold Cross-Validation":
    k_folds = st.sidebar.slider("Number of Folds", 2, 10, 5)

# Step 4: Model Selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose Model", ["K-Nearest Neighbors", "Logistic Regression", "SVM", "Random Forest", "Decision Tree", "Neural Network"])
hyperparameters = {}

# Customize initial hyperparameters based on model choice
if model_choice == "K-Nearest Neighbors":
    hyperparameters['n_neighbors'] = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=hyperparameters['n_neighbors'])
elif model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "SVM":
    model = SVC()
elif model_choice == "Random Forest":
    hyperparameters['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=hyperparameters['n_estimators'])
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "Neural Network":
    hyperparameters['hidden_layer_sizes'] = st.sidebar.slider("Hidden Layer Sizes", 10, 200, 100)
    model = MLPClassifier(hidden_layer_sizes=(hyperparameters['hidden_layer_sizes'],))

# Step 5: Train Model
if st.button("Train Model"):
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Preprocess data if necessary
    if scaling_option == "Standard Scaling":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaling_option == "Min-Max Scaling":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    if validation_type == "Holdout":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    elif validation_type == "K-Fold Cross-Validation":
        kf = StratifiedKFold(n_splits=k_folds)
        accuracies = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        st.write("Average Accuracy:", np.mean(accuracies))

# Step 6: Hyperparameter Tuning
st.sidebar.header("Hyperparameter Tuning")
if st.sidebar.button("Perform Grid Search"):
    param_grid = {'n_neighbors': [3, 5, 7]} if model_choice == "K-Nearest Neighbors" else {}
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=k_folds)
        grid_search.fit(X, y)
        st.write("Best Parameters:", grid_search.best_params_)

# Step 7: Download Model
if st.button("Save Model"):
    joblib.dump(model, "model.pkl")
    st.write("Model saved as model.pkl")

# Step 8: Load and Use Saved Model
uploaded_model = st.file_uploader("Upload a Saved Model", type=["pkl"])
if uploaded_model:
    loaded_model = joblib.load(uploaded_model)
    st.write("Loaded model. Enter data to make predictions.")
