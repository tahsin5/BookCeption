import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('../data/ratings.csv')
books_data = pd.read_csv('../data/books.csv')

books = pd.DataFrame(books_data, columns=['book_id', 'authors', 'title', 'average_rating','ratings_count', 'image_url'])
books_info = pd.merge(books, ratings, on='book_id')

book_ratings = pd.pivot_table(books_info, index='user_id', values='rating', columns='book_id',fill_value=0)
book_rating_matrix = csr_matrix(book_ratings.values)

model_KNN = NearestNeighbors(metric='cosine', algorithm='brute')
model_KNN.fit(book_rating_matrix)

query_index = np.random.choice(book_ratings.shape[0])
distances, indices = model_KNN.kneighbors(book_ratings.iloc[query_index, :].reshape(1,-1), n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i ==0:
        print("Recommendations for {0}:\n".format(book_ratings.index[query_index]))
    else:
        print("{0}: {1} with distance of {2}:".format(i, book_ratings.index[indices.flatten()[i]], distances.flatten()))

transposed = book_ratings.values.T
SVD = TruncatedSVD(n_components=12, random_state=0)
matrix = SVD.fit_transform(transposed)

corr = np.corrcoef(matrix)
corr_input = corr[1]          #corr of input
print(corr_input[corr_input>0.5])