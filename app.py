import streamlit as st
import pandas as pd
from recommendation_engine import get_combined_recommendations, movies_df, knn, indices


# Streamlit app layout
st.title("Movie Recommendation System - Krithiik H")
st.write("""This app recommends movies you'll love based on your pick! 
         Simply choose a movie, and it’ll show you others with a similar vibe and genre!""")

# Limit the number of movies in the dropdown to the top 500 by rating count
top_movies = movies_df.sort_values(by='rating_count', ascending=False).head(500)
movie_title = st.selectbox("Choose a movie title to get recommendations", top_movies['title'].values)

# Movie selection dropdown
#movie_title = st.selectbox("Choose a movie title to get recommendations", movies_df['title'].values)

#movie_title_input = st.text_input("Enter movie title for recommendations")

#if movie_title_input:
    # Find closest match in the movie dataset
#    possible_titles = movies_df[movies_df['title'].str.contains(movie_title_input, case=False)]
#    if not possible_titles.empty:
#        movie_title = possible_titles['title'].iloc[0]
#        recommendations = get_combined_recommendations(movie_title, knn_model=knn)
#        st.write(f"**Top 10 Recommended Movies for {movie_title}:**")
#        st.table(recommendations)
#    else:
#        st.write("No movie found with that title.")

if movie_title:
    recommendations = get_combined_recommendations(movie_title, knn_model=knn)
    st.write("**Top 10 Recommended Movies:**")
    st.table(recommendations)