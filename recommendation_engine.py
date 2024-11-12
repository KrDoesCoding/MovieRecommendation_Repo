import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack 


# Load the movie, ratings, tags, and genome data
movies_df = pd.read_csv(r'C:\KrFiles\My Projects\movierecomm_project\movie.csv', nrows=50000)
ratings_df = pd.read_csv(r'C:\KrFiles\My Projects\movierecomm_project\rating.csv')
tags_df = pd.read_csv(r'C:\KrFiles\My Projects\movierecomm_project\tag.csv')
genome_tags_df = pd.read_csv(r'C:\KrFiles\My Projects\movierecomm_project\genome_tags.csv')
genome_scores_df = pd.read_csv(r'C:\KrFiles\My Projects\movierecomm_project\genome_scores.csv')


### Step 1: Data Preprocessing for Movies, Ratings, and Tags ###

# Process 'genres' column as before
movies_df['genres'] = movies_df['genres'].fillna('').astype(str)
movies_df['genres'] = movies_df['genres'].str.split('|')
movies_df['genres_bag'] = movies_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else 'Unknown')

# Aggregate ratings data by movie
ratings_summary = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
ratings_summary.columns = ['avg_rating', 'rating_count']
movies_df = movies_df.merge(ratings_summary, left_on='movieId', right_index=True, how='left')
movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0)
movies_df['rating_count'] = movies_df['rating_count'].fillna(0)

# Aggregate tags by movie
tags_summary = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
tags_summary.columns = ['movieId', 'tags']
movies_df = movies_df.merge(tags_summary, on='movieId', how='left')
movies_df['tags'] = movies_df['tags'].fillna('')




### Step 2: Merge Genome Tags and Scores ###

# Merge genome tags and scores to create a movie-tag relevance profile
genome_df = pd.merge(genome_scores_df, genome_tags_df, on='tagId', how='inner')

# Aggregate by movieId to create a single string of weighted tags for each movie
# Here, we multiply each tag by its relevance score for a weighted "bag of tags"
genome_summary = genome_df.groupby('movieId', group_keys=False).apply(
    lambda x: ' '.join([f"{row['tag']} " * int(row['relevance'] * 10) for _, row in x.iterrows()])
).reset_index()
genome_summary.columns = ['movieId', 'genome_tags_bag']

# Merge genome tags back to movies_df
movies_df = movies_df.merge(genome_summary, on='movieId', how='left')
movies_df['genome_tags_bag'] = movies_df['genome_tags_bag'].fillna('')




### Step 3: Vectorize Genres, Tags, and Genome Tags ###

# Vectorize genres
genre_vectorizer = CountVectorizer(stop_words='english')
genre_matrix = genre_vectorizer.fit_transform(movies_df['genres_bag'])

# Vectorize user-generated tags
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_tags = tfidf_vectorizer.fit_transform(movies_df['tags'])

# Vectorize genome tags with TF-IDF
tfidf_vectorizer_genome = TfidfVectorizer(stop_words='english')
tfidf_matrix_genome = tfidf_vectorizer_genome.fit_transform(movies_df['genome_tags_bag'])

# Combine all matrices: genres, user tags, and genome tags
combined_matrix = hstack([genre_matrix, tfidf_matrix_tags, tfidf_matrix_genome])

# Initialize NearestNeighbors with cosine distance and fit it to the combined matrix
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11)
knn.fit(combined_matrix)

# Build an index for movie titles
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()




### Step 4: Recommendation Function ###

def get_combined_recommendations(title, knn_model=knn, weight_rating=True):
    idx = indices[title]
    
    # Get the distances and indices for nearest neighbors
    distances, indices_knn = knn_model.kneighbors(combined_matrix[idx], n_neighbors=11)
    
    # Get the movie indices (skip the first result, as it is the movie itself)
    movie_indices = indices_knn[0][1:]
    
    # Get the similar movies and their details
    recommendations = movies_df.iloc[movie_indices][['title', 'avg_rating', 'rating_count']]
    
    if weight_rating:
        recommendations['weighted_score'] = recommendations['avg_rating'] * recommendations['rating_count']
        recommendations = recommendations.sort_values(by='weighted_score', ascending=False)
    
    return recommendations[['title', 'avg_rating', 'rating_count', 'weighted_score']]

