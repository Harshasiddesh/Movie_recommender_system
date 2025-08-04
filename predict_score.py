import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def predict_audience_score(movie_title):
    """
    Predict audience score for a given movie title
    """
    try:
        # Load the dataset
        df = pd.read_csv('Data/Final.csv')
        
        # Check if movie exists
        movie_data = df[df['title'].str.contains(movie_title, case=False, na=False)]
        if len(movie_data) == 0:
            return "Movie not found in database"
        
        # If movie already has an audience score, return it
        if 'audienceScore' in df.columns and not pd.isna(movie_data['audienceScore'].iloc[0]):
            return f"Actual Audience Score: {movie_data['audienceScore'].iloc[0]}"
        
        # For demonstration, return a placeholder prediction
        # In a real implementation, you would train a model on your data
        return "Predicted Audience Score: 75 (placeholder)"
        
    except Exception as e:
        return f"Error: {str(e)}" 