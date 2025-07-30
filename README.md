# NTI-DataScience-Capstone

## Introduction
Our project digs into video game reviews and ratings to find out what players really love (or don't love) about different games.

## Project Overview
We built a system that can predict how users might rate a video game based on different features like graphics quality, soundtrack, story, and more!

### What We Found (Key Findings)
- The most important factors affecting game ratings are:
  - Game Price
  - Game Length (Hours)
- Surprisingly, features like graphics and story quality had less impact than we expected!
- Most mobile games and games for "All Ages" got slightly higher ratings
- Sports and RPG games tend to get better ratings
- Multiplayer games usually score a bit higher

### Data Science Steps We Followed
1. **Problem Definition**: We wanted to predict video game ratings
2. **Data Collection**: Used an SQL Server to fetch the dataset from
3. **Data Understanding**: Looked at different game features
4. **Data Cleaning**: Our data was pretty clean!
5. **Feature Engineering**: Created some cool new features like:
   - Overall Quality Score
   - Review Sentiment Analysis
6. **Data Analysis**: Made lots of graphs to understand patterns
7. **Modeling**: Tried different prediction models

## Our Web App Workflow
Our Streamlit app (`app.py`) works like this:
1. **Dashboard Page**
   - Shows overview of your dataset
   - Displays model results
   - Shows score reports

2. **Upload & Train Page**
   - Upload your CSV file
   - Pick which column to predict
   - Train the model with one click!
   - See how well the model performs

## How to Set Up the Project

### For Windows Users
1. Open Command Prompt
2. Navigate to project folder:
   ```cmd
   cd path\to\project
   ```
3. Run setup:
   ```cmd
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```

### For Linux/Mac Users
1. Open Terminal
2. Navigate to project folder:
   ```bash
   cd path/to/project
   ```
3. Make setup file executable:
   ```bash
   chmod +x setup.sh
   ```
4. Run setup:
   ```bash
   ./setup.sh
   ```

### Running the Web App
After setup, run:
```bash
streamlit run Streamlit/app.py
```

## What We Learned
- Linear Regression worked surprisingly well for our predictions!
- Game price and length are super important for ratings
- People really care about getting good value for their money
- Fancy graphics aren't everything - gameplay length matters more
- Machine learning can help understand what gamers like