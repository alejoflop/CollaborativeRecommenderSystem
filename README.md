from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Calcula similitudes de usuarios y películas una vez
user_sim_matrix = cosine_similarity(train_sparse_matrix, train_sparse_matrix)
movie_sim_matrix = cosine_similarity(train_sparse_matrix.T, train_sparse_matrix.T)

final_data = []

start = datetime.now()

for (user, movie, rating) in zip(train_users, train_movies, train_ratings):
    st = datetime.now()
    
    # Ratings de "movie" por usuarios similares de "user"
    top_sim_users = np.argsort(user_sim_matrix[user])[::-1][1:] # Ignoramos el usuario actual
    top_sim_users_ratings = train_sparse_matrix[top_sim_users, movie].toarray().ravel()
    top_sim_users_ratings = top_sim_users_ratings[top_sim_users_ratings != 0][:5]
    top_sim_users_ratings = np.concatenate([top_sim_users_ratings, [train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings))])
    
    # Ratings por "user" para películas similares de "movie"
    top_sim_movies = np.argsort(movie_sim_matrix[movie])[::-1][1:]
    top_sim_movies_ratings = train_sparse_matrix[user, top_sim_movies].toarray().ravel()
    top_sim_movies_ratings = top_sim_movies_ratings[top_sim_movies_ratings != 0][:5]
    top_sim_movies_ratings = np.concatenate([top_sim_movies_ratings, [train_averages['user'][user]]*(5 - len(top_sim_movies_ratings))])
    
    # Preparar la fila para ser guardada en un DataFrame
    row = [user, movie, train_averages['global']]
    row.extend(top_sim_users_ratings)
    row.extend(top_sim_movies_ratings)
    row.extend([train_averages['user'][user], train_averages['user'][movie], rating])
    final_data.append(row)
    
    count += 1
    if count % 10000 == 0:
        print("Done for {} rows---- {}".format(count, datetime.now() - start))

  Other resource: https://github.com/nishantml/NETFLIX-MOVIE-RECOMMENDATION-SYSTEM/blob/master/Netflix_Movie.ipynb
