import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to fetch movie data from TMDB API
def fetch_movie_data():
    api_key = 'YOUR_API_KEY'  # Replace with your TMDB API key
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={api_key}'
    response = requests.get(url)
    data = response.json()
    return data['results']

# Preprocess the movie data
def preprocess_data(movie_data):
    movie_list = []
    for movie in movie_data:
        title = movie['title']
        genres = ', '.join([genre['name'] for genre in movie['genres']])
        movie_list.append({'title': title, 'genres': genres})
    return movie_list

# Create a matrix of movie genres
def create_genre_matrix(movie_list):
    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform([movie['genres'] for movie in movie_list])
    return genre_matrix

# Calculate cosine similarity matrix
def calculate_cosine_similarity(genre_matrix):
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    return cosine_sim

# Function to get movie recommendations
def get_recommendations(movie_title, cosine_sim_matrix, movie_list):
    idx = [movie['title'] for movie in movie_list].index(movie_title)
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_movies = sim_scores[1:11]
    recommended_movies = [movie_list[movie[0]]['title'] for movie in top_similar_movies]
    return recommended_movies

# Fetch movie data from TMDB API
movie_data = fetch_movie_data()

# Preprocess the movie data
movies_list = preprocess_data(movie_data)

# Create genre matrix
genre_matrix = create_genre_matrix(movies_list)

# Calculate cosine similarity matrix
cosine_sim = calculate_cosine_similarity(genre_matrix)

# Example usage
movie_title = 'The Dark Knight'
genre_input = input("Enter your preferred genre: ")
filtered_movies_list = [movie for movie in movies_list if genre_input.lower() in movie['genres'].lower()]

if len(filtered_movies_list) > 0:
    recommendations = get_recommendations(movie_title, cosine_sim, filtered_movies_list)
    print(f"Recommended movies for '{movie_title}' and genre '{genre_input}':")
    print(recommendations)
else:
    print(f"No movies found for genre '{genre_input}'.")
