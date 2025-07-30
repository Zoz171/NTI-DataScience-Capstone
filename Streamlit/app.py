import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os

# Set page config
st.set_page_config(page_title="ML Model Training App", layout="wide")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = {}

# Model definitions
MODEL_DEFINITIONS = {
    "LG": "Logistic Regression - A linear model for classification that predicts the probability of occurrence of an event.",
    "DT": "Decision Tree Classifier - A tree-structured model that makes decisions based on feature thresholds.",
    "RF": "Random Forest Classifier - An ensemble of decision trees that uses bagging to improve prediction accuracy.",
    "KNN": "K-Nearest Neighbors Classifier - Classifies based on the majority class of K nearest training samples.",
    "SVC": "Support Vector Classifier - Creates a hyperplane that best separates classes with maximum margin.",
    "GNB": "Gaussian Naive Bayes - A probabilistic classifier based on Bayes' theorem with independence assumptions.",
    "GBC": "Gradient Boosting Classifier - An ensemble method that builds trees sequentially to correct previous errors.",
    # Regression models
    "LR": "Linear Regression - A linear approach to modeling the relationship between features and target.",
    "DTR": "Decision Tree Regressor - A tree-structured model for continuous value prediction.",
    "RFR": "Random Forest Regressor - An ensemble of decision trees for robust regression predictions.",
    "KNNR": "K-Nearest Neighbors Regressor - Predicts based on the average of K nearest training samples.",
    "SVR": "Support Vector Regressor - Applies SVM principles to regression tasks.",
    "GBR": "Gradient Boosting Regressor - An ensemble method that combines weak learners for accurate predictions."
}

# Define classification and regression models
def get_models(problem_type):
    models = []
    if problem_type == 'classification':
        models.extend([
            ["LG", LogisticRegression(class_weight='balanced')],
            ["DT", DecisionTreeClassifier()],
            ["RF", RandomForestClassifier()],
            ["KNN", KNeighborsClassifier()],
            ["SVC", SVC()],
            ["GNB", GaussianNB()],
            ["GBC", GradientBoostingClassifier()]
        ])
    else:  # regression
        models.extend([
            ["LR", LinearRegression()],
            ["DTR", DecisionTreeRegressor()],
            ["RFR", RandomForestRegressor()],
            ["KNNR", KNeighborsRegressor()],
            ["SVR", SVR()],
            ["GBR", GradientBoostingRegressor()]
        ])
    return models

# Define parameter grids for GridSearchCV
def get_param_grid(model_name):
    param_grids = {
        "LG": {
            "LG__C": [1, 5, 10],
        },
        "DT": {
            "DT__max_depth": [3, 6, 9, 12],
            "DT__min_samples_split": [2, 4, 6],
            "DT__min_samples_leaf": [1, 2, 4],
            "DT__max_features": [None, 2, 4],
            "DT__random_state": [0, 7, 42]
        },
        "RF": {
            "RF__n_estimators": [100, 200, 300],
            "RF__max_depth": [3, 6, 9],
            "RF__min_samples_split": [2, 4],
            "RF__random_state": [42]
        },
        "KNN": {
            "KNN__n_neighbors": [3, 5, 7, 9],
            "KNN__weights": ['uniform', 'distance']
        },
        "SVC": {
            "SVC__C": [0.1, 1, 10],
            "SVC__kernel": ['rbf', 'linear'],
            "SVC__random_state": [42]
        },
        "GNB": {},  # GaussianNB has no hyperparameters to tune
        "GBC": {
            "GBC__n_estimators": [100, 200],
            "GBC__learning_rate": [0.01, 0.1],
            "GBC__max_depth": [3, 5],
            "GBC__random_state": [42]
        },
        # Regression models
        "LR": {},  # LinearRegression has no hyperparameters to tune
        "DTR": {
            "DTR__max_depth": [3, 6, 9, 12],
            "DTR__min_samples_split": [2, 4, 6],
            "DTR__min_samples_leaf": [1, 2, 4],
            "DTR__random_state": [42]
        },
        "RFR": {
            "RFR__n_estimators": [100, 200, 300],
            "RFR__max_depth": [3, 6, 9],
            "RFR__random_state": [42]
        },
        "KNNR": {
            "KNNR__n_neighbors": [3, 5, 7, 9],
            "KNNR__weights": ['uniform', 'distance']
        },
        "SVR": {
            "SVR__C": [0.1, 1, 10],
            "SVR__kernel": ['rbf', 'linear']
        },
        "GBR": {
            "GBR__n_estimators": [100, 200],
            "GBR__learning_rate": [0.01, 0.1],
            "GBR__max_depth": [3, 5],
            "GBR__random_state": [42]
        }
    }
    return param_grids.get(model_name, {})

def train_model(X_train, X_test, y_train, y_test, model_name, model, problem_type):
    # Create pipeline
    steps = [
        ("Scalar", RobustScaler()),
        (model_name, model)
    ]
    pipeline = Pipeline(steps=steps)
    
    # Get parameter grid
    param_grid = get_param_grid(model_name)
    
    # Set scoring metric based on problem type
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    # Make predictions
    y_pred = grid_search.best_estimator_.predict(X_test)
    
    # Create results dictionary
    results = {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'train_score': grid_search.cv_results_['mean_train_score'][grid_search.best_index_],
        'test_score': grid_search.cv_results_['mean_test_score'][grid_search.best_index_],
        'y_pred': y_pred
    }
    
    return results

def dashboard():
    st.title("Dashboard")
    
    if st.session_state.data is not None:
        st.header("Dataset Preview")
        st.dataframe(st.session_state.data.head())
        
        if st.session_state.training_results:
            st.header("Training Results")
            
            # Add a save button for each model at the top
            col1, col2 = st.columns([3, 1])
            with col1:
                model_to_save = st.selectbox("Select model to save", list(st.session_state.training_results.keys()))
            with col2:
                if st.button("Save Selected Model"):
                    save_path = f"{model_to_save}_model.pkl"
                    with open(save_path, 'wb') as f:
                        pickle.dump(st.session_state.training_results[model_to_save]['model'], f)
                    st.success(f"Model saved as {save_path}")
            
            for model_name, results in st.session_state.training_results.items():
                with st.expander(f"{model_name} Results"):
                    # Show model definition
                    st.markdown(f"**Model Definition:**")
                    st.info(MODEL_DEFINITIONS.get(model_name, "Definition not available"))
                    
                    # Show model performance
                    st.markdown("**Model Performance:**")
                    st.write(f"Training Score: {results['train_score']:.4f}")
                    st.write(f"Testing Score: {results['test_score']:.4f}")
                    
                    # Show metrics based on the model's problem type
                    if results.get('problem_type') == 'classification':
                        if 'classification_report' in results:
                            st.markdown("**Classification Report:**")
                            st.text(results['classification_report'])
                        
                        # Display confusion matrix
                        st.markdown("**Confusion Matrix:**")
                        if 'confusion_matrix' in results:
                            # Create a new figure for each display
                            fig, ax = plt.subplots(figsize=(8, 6))
                            disp = ConfusionMatrixDisplay(confusion_matrix=results['confusion_matrix'])
                            disp.plot(ax=ax)
                            plt.title(f"{model_name} Confusion Matrix")
                            st.pyplot(fig)
                            plt.close(fig)  # Close the figure to free memory
                    elif results.get('problem_type') == 'regression':
                        st.markdown("**Regression Metrics:**")
                        if all(key in results for key in ['rmse', 'mae', 'r2']):
                            st.write(f"RMSE: {results['rmse']:.4f}")
                            st.write(f"MAE: {results['mae']:.4f}")
                            st.write(f"R2 Score: {results['r2']:.4f}")
                        else:
                            st.warning("Some regression metrics are missing for this model.")
    else:
        st.info("Please upload a dataset in the Training page to get started.")

def training():
    st.title("Model Training")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        
        # Select target column
        target_column = st.selectbox("Select target column", data.columns)
        st.session_state.target_column = target_column
        
        # Determine problem type
        unique_values = data[target_column].nunique()
        if unique_values <= 10:  # Assuming classification if 10 or fewer unique values
            problem_type = "classification"
        else:
            problem_type = "regression"
            
        # Clear training results if problem type changes
        if 'problem_type' in st.session_state and st.session_state.problem_type != problem_type:
            st.session_state.training_results = {}
            
        st.session_state.problem_type = problem_type
        
        st.write(f"Detected problem type: {problem_type}")
        
        # Get available models
        models = get_models(problem_type)
        model_names = [m[0] for m in models]
        
        # Model selection
        selected_models = st.multiselect("Select models to train", model_names)
        
        if selected_models:
            # Split data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if st.button("Train Models"):
                for model_name in selected_models:
                    st.subheader(f"Training {model_name}")
                    
                    # Get model object
                    model = [m[1] for m in models if m[0] == model_name][0]
                    
                    # Train model
                    results = train_model(X_train, X_test, y_train, y_test, model_name, model, problem_type)
                    
                    # Store results in session state
                    if problem_type == 'classification':
                        try:
                            # Create and store confusion matrix
                            cm = confusion_matrix(y_test, results['y_pred'], normalize='true')
                            results['confusion_matrix'] = cm
                            
                            # Create and store classification report
                            results['classification_report'] = classification_report(y_test, results['y_pred'])
                            
                            # Create confusion matrix visualization
                            fig, ax = plt.subplots(figsize=(8, 6))
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                            disp.plot(ax=ax)
                            plt.title(f"{model_name} Confusion Matrix")
                            results['confusion_matrix_fig'] = fig
                            plt.close(fig)  # Close the figure to free memory
                        except Exception as e:
                            st.error(f"Error creating classification metrics: {str(e)}")
                    else:  # regression
                        try:
                            results['rmse'] = np.sqrt(mean_squared_error(y_test, results['y_pred']))
                            results['mae'] = mean_absolute_error(y_test, results['y_pred'])
                            results['r2'] = r2_score(y_test, results['y_pred'])
                        except Exception as e:
                            st.error(f"Error creating regression metrics: {str(e)}")

                    # Add problem type to results for proper display in dashboard
                    results['problem_type'] = problem_type
                    
                    st.session_state.training_results[model_name] = results
                    
                    # Display results
                    st.write("Best Parameters:", results['best_params'])
                    st.write("Training Score:", results['train_score'])
                    st.write("Testing Score:", results['test_score'])
                    
                    if problem_type == 'classification':
                        # Classification metrics
                        st.text("Classification Report:")
                        st.text(results['classification_report'])
                        st.pyplot(results['confusion_matrix_fig'])
                    else:
                        # Regression metrics
                        st.write("RMSE:", results['rmse'])
                        st.write("MAE:", results['mae'])
                        st.write("R2 Score:", results['r2'])

# Main app navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Training"])
    
    if page == "Dashboard":
        dashboard()
    else:
        training()

if __name__ == "__main__":
    main()
