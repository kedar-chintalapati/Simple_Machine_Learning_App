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

# Step 1: App Title
st.title("Machine Learning Pipeline App")

# Step 2: Data Upload/Entry
st.header("Step 1: Data Upload")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
data = None

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Let the user select the target column and feature columns
    st.subheader("Select Columns for Model")
    target_column = st.selectbox("Select the target column (output)", data.columns)
    feature_columns = st.multiselect("Select feature columns (input)", data.columns.drop(target_column))

    # Ensure features and target are selected before proceeding
    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]
    else:
        st.warning("Please select both target and feature columns to proceed.")
else:
    st.info("Please upload a CSV file to start.")

# Step 3: Preprocessing Options
if data is not None and target_column and feature_columns:
    st.header("Step 2: Preprocessing")
    scaling_option = st.selectbox("Choose a scaling method", ["None", "Standard Scaling", "Min-Max Scaling"])

    # Apply scaling based on user choice
    if scaling_option == "Standard Scaling":
        scaler = StandardScaler()
    elif scaling_option == "Min-Max Scaling":
        scaler = MinMaxScaler()
    else:
        scaler = None

# Step 4: Data Splitting Options
    st.header("Step 3: Data Splitting")
    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
    validation_type = st.selectbox("Validation Type", ["Holdout", "K-Fold Cross-Validation"])
    k_folds = 1
    if validation_type == "K-Fold Cross-Validation":
        k_folds = st.slider("Number of Folds", 2, 10, 5)

# Step 5: Model Selection and Initial Hyperparameters
    st.header("Step 4: Model Selection")
    model_choice = st.selectbox("Choose Model", ["K-Nearest Neighbors", "Logistic Regression", "SVM", "Random Forest", "Decision Tree", "Neural Network"])
    
    # Initialize model with default or user-selected hyperparameters
    if model_choice == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "SVM":
        model = SVC()
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Neural Network":
        hidden_layer_sizes = st.slider("Hidden Layer Sizes", 10, 200, 100)
        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,))

# Step 6: Train Model
    st.header("Step 5: Training")
    if st.button("Train Model"):
        if scaler:
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

# Step 7: Hyperparameter Tuning
    st.header("Step 6: Hyperparameter Tuning")
    if st.button("Perform Grid Search"):
        param_grid = {}
        if model_choice == "K-Nearest Neighbors":
            param_grid = {'n_neighbors': [3, 5, 7, 10]}
        elif model_choice == "Random Forest":
            param_grid = {'n_estimators': [50, 100, 150]}
        elif model_choice == "Neural Network":
            param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,)]}
        
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=k_folds)
            grid_search.fit(X, y)
            st.write("Best Parameters:", grid_search.best_params_)
        else:
            st.info("No hyperparameters available for tuning with this model.")

# Step 8: Download Model
    st.header("Step 7: Save and Load Model")
    if st.button("Save Model"):
        joblib.dump(model, "model.pkl")
        st.write("Model saved as model.pkl")
        st.download_button("Download Model", data=open("model.pkl", "rb"), file_name="model.pkl")

    uploaded_model = st.file_uploader("Upload a Saved Model", type=["pkl"])
    if uploaded_model:
        loaded_model = joblib.load(uploaded_model)
        st.write("Loaded model. Enter data to make predictions.")
