import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(movie_title, n_recommendations=5):
    """
    Get movie recommendations based on content similarity
    """
    try:
        # Load the dataset
        df = pd.read_csv('Data/Final.csv')
        
        # Create TF-IDF vectorizer for text features
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Combine text features (assuming you have columns like 'genre', 'director', etc.)
        text_features = df['genre'].fillna('') + ' ' + df['director'].fillna('')
        
        # Create TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(text_features)
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Find the movie index
        movie_idx = df[df['title'].str.contains(movie_title, case=False, na=False)].index
        if len(movie_idx) == 0:
            return ["Movie not found in database"]
        
        movie_idx = movie_idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommended movies
        recommendations = df['title'].iloc[movie_indices].tolist()
        return recommendations
        
    except Exception as e:
        return [f"Error: {str(e)}"] 