<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Movie Recommendation System</title>
</head>
<body>
<h1>Movie Recommendation System</h1>

<p>This project is a <strong>Movie Recommendation System</strong> built using <strong>K-Nearest Neighbors (K-NN)</strong> to suggest movies based on their genres, user tags, and genome tags. The system calculates similarity scores and generates recommendations based on movie metadata, user-generated tags, and user ratings. The project also includes an interactive <strong>Streamlit web application</strong> that allows users to select a movie title and receive personalized movie recommendations.</p>

<hr>

<h2 id="overview">Overview</h2>
<p>The Movie Recommendation System leverages the K-NN algorithm to provide tailored movie recommendations by analyzing a combination of genres, user tags, and genome tags. This content-based approach avoids collaborative filtering, focusing solely on movie attributes and similarity calculations to create recommendations.</p>

<h2 id="features">Features</h2>
  <ul>
        <li><strong>K-Nearest Neighbors (K-NN) Based Recommendations</strong>: Uses K-NN to identify the closest movies based on combined similarity metrics.</li>
        <li><strong>Integrated Streamlit App</strong>: Provides an intuitive UI for users to select a movie and view top recommendations.</li>
        <li><strong>Data Preprocessing</strong>: Cleans and preprocesses movie data, aggregating user ratings, genres, and tags.</li>
  </ul>


<h2 id="how-it-works">How It Works</h2>
    <ol>
        <li><strong>Data Preprocessing</strong>: Genres are tokenized and vectorized, user tags and genome tags are preprocessed and vectorized using TF-IDF to create meaningful representations.</li>
        <li><strong>Combining Features</strong>: Genres, user tags, and genome tags are combined into a single matrix.</li>
        <li><strong>K-NN Model</strong>: The K-NN model identifies the 10 nearest movies based on the combined matrix.</li>
        <li><strong>Recommendation Function</strong>: Given a movie title, the model returns the top 10 similar movies based on similarity scores, with optional weighting for user ratings.</li>
        <li><strong>Streamlit App</strong>: Users select a movie from a dropdown and receive the top 10 recommendations based on the model’s output.</li>
    </ol>


<h2 id="usage">Usage</h2>
    <ol>
        <li><strong>Run the Streamlit app</strong>:
            <pre><code>streamlit run app.py</code></pre>
        </li>
        <li><strong>Select a movie</strong> from the dropdown menu to get recommendations.</li>
    </ol>

</body>
</html>
