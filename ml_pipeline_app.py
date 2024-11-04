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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import joblib

# Step 1: App Title
st.title("Machine Learning Laboratory")

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

# Step 5: Model Selection and Customization for Neural Networks
    st.header("Step 4: Model Selection")
    model_choice = st.selectbox("Choose Model", ["K-Nearest Neighbors", "Logistic Regression", "SVM", "Random Forest", "Decision Tree", "Neural Network"])

    # Initialize model with user-selected hyperparameters and settings
    if model_choice == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "SVM":
        model = SVC(probability=True)
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Neural Network":
        st.subheader("Neural Network Configuration")
        num_layers = st.slider("Number of Hidden Layers", 1, 5, 1)
        hidden_layer_sizes = []
        activation_functions = []

        for i in range(num_layers):
            layer_size = st.number_input(f"Size of Layer {i + 1}", min_value=1, max_value=200, value=100, step=10)
            activation_function = st.selectbox(f"Activation Function for Layer {i + 1}", ["relu", "tanh", "logistic"], key=f"activation_{i}")
            hidden_layer_sizes.append(layer_size)
            activation_functions.append(activation_function)

        output_activation = st.selectbox("Output Layer Activation Function", ["identity", "logistic", "tanh", "relu"])
        model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation_functions[0])

# Step 6: Train Model and Save Train/Test Split to Session State
    st.header("Step 5: Training")
    if st.button("Train Model"):
        if scaler:
            X = scaler.fit_transform(X)

        if validation_type == "Holdout":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['model'] = model
            st.write("Model trained successfully!")
        elif validation_type == "K-Fold Cross-Validation":
            kf = StratifiedKFold(n_splits=k_folds)
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['model'] = model
            st.write("K-Fold Cross-Validation training completed.")

# Step 7: Evaluation Metrics with Buttons
    st.header("Step 6: Evaluation")
    if 'X_test' in st.session_state and 'model' in st.session_state:
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        model = st.session_state['model']
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        if st.button("Train Accuracy"):
            st.write("Train Accuracy:", accuracy_score(y_train, y_pred_train))

        if st.button("Test Accuracy"):
            st.write("Test Accuracy:", accuracy_score(y_test, y_pred_test))

        if st.button("Test F1 Score"):
            st.write("Test F1 Score:", f1_score(y_test, y_pred_test, average='weighted'))

        if st.button("Test AUC"):
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
                st.write("Test AUC:", auc)
            except AttributeError:
                st.warning("AUC not available for this model.")

# Step 8: Hyperparameter Tuning
    st.header("Step 7: Hyperparameter Tuning")
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

# Step 9: Save and Load Model
    st.header("Step 8: Save and Load Model")
    if st.button("Save Model"):
        joblib.dump(model, "model.pkl")
        st.write("Model saved as model.pkl")
        st.download_button("Download Model", data=open("model.pkl", "rb"), file_name="model.pkl")

    uploaded_model = st.file_uploader("Upload a Saved Model", type=["pkl"])
    if uploaded_model:
        loaded_model = joblib.load(uploaded_model)
        st.session_state['model'] = loaded_model
        st.write("Loaded model. Enter data to make predictions.")

# Step 10: Single-Sample Prediction
    st.header("Step 9: Single Sample Prediction")
    st.write("Enter data for a single sample below:")

    single_sample = []
    for feature in feature_columns:
        value = st.number_input(f"Enter {feature} value", value=0.0)
        single_sample.append(value)

    if st.button("Predict on Single Sample"):
        if 'model' in st.session_state:
            sample = np.array(single_sample).reshape(1, -1)
            if scaler:
                sample = scaler.transform(sample)
            prediction = st.session_state['model'].predict(sample)
            st.write("Prediction for entered sample:", prediction[0])
