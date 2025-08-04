
# Content-Based Movie Recommender System & Audience Score Prediction

This repository contains a **Content-Based Movie Recommender System** and an **Audience Score Prediction** model, designed to recommend movies based on features like genre, director, writer, language, and review sentiment, and to predict missing audience scores using machine learning.

## 🔍 Project Description

- **Movie Recommender System**: Uses TF-IDF and cosine similarity on combined features (genre, director, writer, language, and cleaned review text) to recommend similar movies to a user-specified title.
- **Audience Score Prediction**: Implements a scikit-learn pipeline (with feature engineering and Random Forest regression) to predict missing audience scores for movies.

## 🗂 Project Structure
('''
Movie_Recommendation/

├── app.py              # Flask application entry point
├── templates/          # HTML templates
│ ├── index.html        # Main landing page
│ └── results.html      # Results display page
├── Data/
│ ├── movie.csv         # Raw movie metadata
│ └── credits.csv       # Raw review metadata
│ └── Final.csv         # Cleaned and merged dataset
├── models/
│ ├── EDA.ipynb         # Exploratory Data Analysis notebook
│ ├── recommend.py      # Recommendation module
│ └── predict_score.py  # Audience score prediction module
├── requirements.txt    # Python dependencies
└── README.md           # This file
''')
## 📁 Data Source

- **Kaggle Dataset**: [Clapper: Massive Rotten Tomatoes Movies and Reviews](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/data)
- **Movie Dataset**: `Data/movie.csv` (~1.4M rows, 16 columns per movie)
- **Review Dataset**: `Data/credits.csv` (~14M rows, 11 columns per review)
- **Final Merged Data**: `Data/Final.csv` (cleaned, feature-engineered for analysis and modeling)

---

## ✨ Features

- **Content-based movie recommendations**: Find movies similar to your favorite titles, leveraging genre, director, writer, language, and review sentiment.
- **Audience score prediction**: If a movie’s audience score is missing, predict it using a combination of content and review features.
- **Exploratory data analysis**: Extensive notebooks and scripts for cleaning, analyzing, and merging large-scale movie and review data.
- **Web interface**: Easy-to-use browser interface to query movies and reviews.

---

## 🧩 Project Components

- **EDA.ipynb**: Data cleaning, exploratory analysis, and feature engineering for movie and review data.
- **predict_score.py**: Wrapper for scikit-learn-based audience score prediction using Random Forest regression.
- **recommend.py**: Wrapper for generating movie recommendations using TF-IDF and cosine similarity.
- **Final.csv**: Cleaned, merged dataset of movies and aggregated reviews.
- **app.py**: Flask web application that connects the recommender and score predictor to a user-friendly GUI.
- **HTML templates**: Simple, expandable front-end interface.

---

---

## 🔍 How Does It Work?

- **Data Cleaning**: In `EDA.ipynb`, movie and review data are merged, missing values are handled, and features like genre, director, writer, language, and review text are cleaned and combined.
- **Recommendations**: The recommender (`recommend.py`) uses TF-IDF vectorization on movie features and computes cosine similarity to find the most similar movies.
- **Score Prediction**: The score predictor (`predict_score.py`) uses a Random Forest regressor trained on available audience scores to predict scores for movies where this information is missing.

---

## 🧮 Model Details

- **Recommendation**: Based on content similarity using TF-IDF vectorization and cosine similarity over movie features.
- **Scoring**: Random Forest regression on engineered movie and aggregated review features (median score, sentiment, etc.).
- **Feature Engineering**: Text preprocessing, normalization of review scores, aggregation of reviews per movie, and categorical encoding.
