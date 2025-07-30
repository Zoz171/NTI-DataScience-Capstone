import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from textblob import TextBlob

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Train Models", "Make Predictions"])

def load_and_process_data(file):
    df = pd.read_csv(file)
    return df

def feature_engineering(df):
    df_copy = df.copy()
    
    # Overall Quality calculation
    sound_and_story_dict = {
        'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3
    }
    graphics_dict = {
        'Low': 0, 'Medium': 1, 'High': 2, 'Ultra': 3
    }
    
    def labeling(row):
        dict_ = {0: 'Poor', 1: 'Average', 2: 'Good', 3: 'Excellent'}
        return dict_.get(np.ceil(row), 0)
    
    df_copy['Overall Quality'] = ((df_copy['Graphics Quality'].map(graphics_dict) + 
                                 df_copy['Soundtrack Quality'].map(sound_and_story_dict) + 
                                 df_copy['Story Quality'].map(sound_and_story_dict)) / 3).apply(labeling)
    
    # Sentiment Analysis
    def get_sentiment_polarity(text):
        if isinstance(text, str):
            return TextBlob(text).sentiment.polarity
        return 0.0
    
    df_copy['Review_Sentiment'] = df_copy['User Review Text'].apply(get_sentiment_polarity)
    
    return df_copy

def create_visualizations(df):
    # User Ratings Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='User Rating', kde=True, ax=ax1)
    plt.title('Distribution of User Ratings')
    st.pyplot(fig1)
    
    # Price vs Ratings
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df[:100], x='User Rating', y='Price', ax=ax2)
    plt.title('User Ratings vs Price')
    st.pyplot(fig2)
    
    # Games by Platform
    platform_counts = df['Platform'].value_counts()
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Games by Platform')
    st.pyplot(fig3)
    
    # Top 5 Games
    top_games = df.nlargest(5, 'User Rating')
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_games, x='User Rating', y='Game Title', ax=ax4)
    plt.title('Top 5 Games by User Rating')
    st.pyplot(fig4)


    top_5_game_titles = top_games['Game Title'].tolist()
    df_top_games_filtered = df[df['Game Title'].isin(top_5_game_titles)].copy()
    df_top_games_filtered = df_top_games_filtered[(df_top_games_filtered['Release Year'] >= 2019) & (df_top_games_filtered['Release Year'] <= 2023)].copy()
    fig5, ax5 = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df_top_games_filtered, x='Release Year', y='User Rating', hue='Game Title', ax=ax5)
    plt.title('User Rating Trend of Top 5 Games Over Time (2019-2023)')
    plt.xlabel('Release Year')
    plt.ylabel('User Rating')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Game Title')
    plt.tight_layout()
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(15,7))
    sns.countplot(data=df, x='Genre', hue='Age Group Targeted', ax=ax6)
    plt.title('Popular Genres for an Age Group')
    plt.legend(title = "Age Group", loc = "upper left", bbox_to_anchor = (1,1))
    st.pyplot(fig6)


def train_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': (LinearRegression(), {'fit_intercept': [True, False]}),
        'Decision Tree': (DecisionTreeRegressor(random_state=42), {'max_depth': [3, 5, 7]}),
        'XGBoost': (XGBRegressor(random_state=42), {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [3]}),
        # 'SVR': (SVR(), {'C': [1.0], 'kernel': ['linear']})
    }

    # Sonnet    
    # results = {}
    # for name, model in models.items():
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     mse = mean_squared_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)
    #     results[name] = {
    #         'model': model,
    #         'MSE': mse,
    #         'R2': r2,
    #         'Train Score': model.score(X_train, y_train),
    #         'Test Score': model.score(X_test, y_test)
    #     }

    results = {}

    for model_name, (model, param_grid) in models.items():
        print(f"Tuning {model_name}...")

        # Perform grid search
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Make predictions with the best model
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store the results with the best model
        results[model_name] = {
            'model': best_model,  # Store the best model, not the original model
            'best_params': grid_search.best_params_,
            'MSE': mse,
            'R2': r2,
            'Train Score': best_model.score(X_train, y_train),
            'Test Score': best_model.score(X_test, y_test)
        }
        print(f"Finished tuning {model_name}")
    
    return results

# --- Page 1: Dashboard ---
if page == "Dashboard":
    st.title("Video Game Rating Analysis Dashboard")
    
    st.header("Project Overview")
    st.write("""
    This project aims to develop a predictive model that can accurately estimate the user rating of a video game based on its various features.
    By analyzing characteristics such as price, platform, genre, multiplayer capabilities, game length, graphics, soundtrack, story quality, and other relevant attributes,
    we aim to build a robust model that can provide insights into what factors influence user satisfaction and predict potential ratings for new or existing games.
    """)
    
    # Load and display data
    uploaded_file = st.file_uploader("Upload video_game_reviews.csv", type=["csv"])
    
    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
        # Only load data if a new file is uploaded
        df = load_and_process_data(uploaded_file)
        df_processed = feature_engineering(df)
        st.session_state.df = df
        st.session_state.df_processed = df_processed
        st.session_state.uploaded_file_name = uploaded_file.name
    
    # Display data if it exists in session state
    if st.session_state.df is not None:
        st.subheader("Dataset Overview")
        st.write(st.session_state.df.head())
        
        st.subheader("Exploratory Data Analysis")
        create_visualizations(st.session_state.df)

# --- Page 2: Train Models ---
elif page == "Train Models":
    st.title("Model Training")
    
    if st.session_state.df is None:
        st.warning("Please upload data in the Dashboard page first!")
    else:
        st.subheader("Feature Engineering & Model Training")
        
        if st.button("Start Training"):
            with st.spinner("Processing data and training models..."):
                # Use already processed data from session state
                df_processed = st.session_state.df_processed
                
                # Prepare data for modeling
                X = df_processed.drop(['User Rating', 'Game Title', 'User Review Text'], axis=1)
                y = df_processed['User Rating']
                
                # Encode categorical features
                categorical_features = X.select_dtypes(include=['object']).columns
                label_encoders = {}
                for feature in categorical_features:
                    le = LabelEncoder()
                    X[feature] = le.fit_transform(X[feature])
                    label_encoders[feature] = le
                st.session_state.label_encoders = label_encoders
                
                # Scale numerical features
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                st.session_state.preprocessor = scaler
                
                # Train models
                results = train_models(X, y)
                st.session_state.models = results
                
                # Store results in session state
                st.session_state.results = results

        # Display results (outside of training button block)
        if 'results' in st.session_state:
            results_df = pd.DataFrame({
                model_name: {
                    'Best Parameters': str(scores['best_params']),
                    'MSE': scores['MSE'],
                    'R2': scores['R2'],
                    'Train Score': scores['Train Score'],
                    'Test Score': scores['Test Score']
                }
                for model_name, scores in st.session_state.results.items()
            }).T
            
            st.subheader("Model Performance Comparison")
            st.write(results_df)
            
        # Save model buttons (outside of the training button's if block)
        if 'results' in st.session_state:
            st.subheader("Save Models")
            cols = st.columns(len(st.session_state.results))
            for idx, (model_name, scores) in enumerate(st.session_state.results.items()):
                with cols[idx]:
                    if st.button(f"Save {model_name}"):
                        # print(scores['model'])
                        model = scores['model']
                        joblib.dump(model, f'{model_name.lower().replace(" ", "_")}.pkl')
                        st.success(f"{model_name} saved successfully!")

# --- Page 3: Make Predictions ---
elif page == "Make Predictions":
    st.title("Make Predictions")
    
    uploaded_model = st.file_uploader("Upload trained model (.pkl)", type=['pkl'])
    
    if uploaded_model:
        model = joblib.load(uploaded_model)
        st.success("Model loaded successfully!")
        
        st.subheader("Enter Game Details")
        
        # Create input form with four columns for better organization
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            platform = st.selectbox("Platform", ['PC', 'PlayStation', 'Xbox', 'Nintendo Switch', 'Mobile'], key="platform_select")
            age_group = st.selectbox("Age Group Targeted", ['Kids', 'Teens', 'Adults', 'All Ages'], key="age_group_select")
            price = st.number_input("Price", min_value=0.0, max_value=100.0, value=29.99, key="price_input")
            # graphics = st.selectbox("Graphics Quality", ['Low', 'Medium', 'High', 'Ultra'], key="graphics_select")
            special_device = st.selectbox("Requires Special Device", ['Yes', 'No'], key="special_device_select")
            review_sentiment = st.number_input("Review Sentiment", min_value=-1.0, max_value=1.0, value=0.033333, step=0.1, key="review_sentiment_input")
            
        with col2:
            developer = st.text_input("Developer", key="developer_input")
            publisher = st.text_input("Publisher", key="publisher_input")
            release_year = st.number_input("Release Year", min_value=1970, max_value=2025, value=2023, key="release_year_input")
            genre = st.selectbox("Genre", ['Action', 'Adventure', 'RPG', 'Sports', 'Strategy'], key="genre_select")
            
        with col3:
            multiplayer = st.selectbox("Multiplayer", ['Yes', 'No'], key="multiplayer_select")
            game_length = st.number_input("Game Length (Hours)", min_value=1, max_value=200, value=20, key="game_length_input")
            min_players = st.number_input("Min Number of Players", min_value=1, max_value=10, value=1, key="min_players_input")
            game_mode = st.selectbox("Game Mode", ['Online', 'Offline', 'Both'], key="game_mode_select")
            
        with col4:
            graphics = st.selectbox("Graphics Quality", ['Low', 'Medium', 'High', 'Ultra'], key="graphics_select")
            soundtrack = st.selectbox("Soundtrack Quality", ['Poor', 'Average', 'Good', 'Excellent'], key="soundtrack_select")
            story = st.selectbox("Story Quality", ['Poor', 'Average', 'Good', 'Excellent'], key="story_select")
            overall_quality = st.selectbox("Overall Quality", ['Poor', 'Average', 'Good', 'Excellent'], key="overall_quality_select")
        
        if st.button("Predict Rating"):
            # Prepare input data
            input_data = pd.DataFrame({
                'Age Group Targeted': [age_group],
                'Price': [price],
                'Platform': [platform],
                'Requires Special Device': [special_device],
                'Developer': [developer],
                'Publisher': [publisher],
                'Release Year': [release_year],
                'Genre': [genre],
                'Multiplayer': [multiplayer],
                'Game Length (Hours)': [game_length],
                'Graphics Quality': [graphics],
                'Soundtrack Quality': [soundtrack],
                'Story Quality': [story],
                'Game Mode': [game_mode],
                'Min Number of Players': [min_players],
                'Overall Quality': [overall_quality],
                'Review_Sentiment': [review_sentiment]
            })
            
            # Process input using saved encoders and scaler
            if st.session_state.label_encoders and st.session_state.preprocessor:
                for column, encoder in st.session_state.label_encoders.items():
                    if column in input_data.columns:
                        input_data[column] = encoder.transform(input_data[column])
                
                input_data = st.session_state.preprocessor.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_data)
                st.success(f"Predicted User Rating: {prediction[0]:.2f}/50")
