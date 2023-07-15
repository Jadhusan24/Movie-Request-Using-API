import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fetch_movie_data():
    # Dummy data for movie API response
    dummy_data = {
        'results': [
            {'title': 'Movie A', 'genres': ['Action', 'Adventure']},
            {'title': 'Movie B', 'genres': ['Drama', 'Romance']},
            {'title': 'Movie C', 'genres': ['Action', 'Comedy']},
            {'title': 'Movie D', 'genres': ['Animation', 'Adventure']},
            {'title': 'Movie E', 'genres': ['Thriller', 'Horror']},
            {'title': 'Movie F', 'genres': ['Comedy', 'Romance']},
            {'title': 'Movie G', 'genres': ['Action', 'Drama']},
            {'title': 'Movie H', 'genres': ['Adventure', 'Sci-Fi']},
            {'title': 'Movie I', 'genres': ['Comedy', 'Family']},
            {'title': 'Movie J', 'genres': ['Drama', 'Crime']}
        ]
    }
    return dummy_data['results']

def preprocess_data(movie_data):
    movie_list = []
    for movie in movie_data:
        title = movie['title']
        genres = ', '.join(movie['genres'])
        movie_list.append({'title': title, 'genres': genres})
    return movie_list

# Create a matrix of movie genres
def create_genre_matrix(movie_list):
    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform([movie['genres'] for movie in movie_list])
    return genre_matrix

def calculate_cosine_similarity(genre_matrix):
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    return cosine_sim

def get_recommendations(movie_title, cosine_sim_matrix, movie_list):
    idx = [movie['title'] for movie in movie_list].index(movie_title)
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_movies = sim_scores[1:11]
    recommended_movies = [movie_list[movie[0]]['title'] for movie in top_similar_movies if movie[0] < len(movie_list)]
    return recommended_movies

movie_data = fetch_movie_data()

movies_list = preprocess_data(movie_data)

genre_matrix = create_genre_matrix(movies_list)

cosine_sim = calculate_cosine_similarity(genre_matrix)

movie_title = 'Movie A'
try:
    genre_input = input("Enter your genre: ")
except KeyboardInterrupt:
    print("\n Bye user.")
    exit()

filtered_movies_list = [movie for movie in movies_list if genre_input.lower() in movie['genres'].lower()]

if len(filtered_movies_list) > 0:
    recommendations = get_recommendations(movie_title, cosine_sim, filtered_movies_list)
    print(f"Recommended movies for '{movie_title}' and genre '{genre_input}':")
    print(recommendations)
else:
    print(f"No movies found for genre '{genre_input}'.")
