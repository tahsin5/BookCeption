import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('../data/ratings.csv')
books_data = pd.read_csv('../data/books.csv')
max = books_data.sort_values(by=['ratings_count'])
#plt.plot(books_data['ratings_count'])
books_data = books_data[(books_data.ratings_count >= 15000)]

books = pd.DataFrame(books_data, columns=['book_id', 'authors', 'title', 'average_rating','ratings_count', 'image_url'])
books_info = pd.merge(books, ratings, on='book_id')

book_ratings = pd.pivot_table(books_info, index='user_id', values='rating', columns='book_id',fill_value=0)
#book_rating_matrix = csr_matrix(book_ratings.values)

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(book_ratings, test_size=0.25)
ptrain = train_data.pivot(index='user_id', columns='book_id', values='rating')
ptest = test_data.pivot(index='user_id', columns='book_id', values='rating')

train_user_index = pd.DataFrame(ptrain.index)
train_books_index = pd.DataFrame(ptrain.columns)

ptest = test_data.pivot(index='user_id', columns='book_id', values='rating')
R = ptrain.fillna(0.0).copy().values
T = ptest.fillna(0.0).copy().values

I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

def prediction(P,Q):
    return np.dot(P.T,Q)

lmbda = 0.1 # L2 penalty Regularization weight (Lambda)
k = 20  # number of the latent features
m, n = R.shape  # Number of users and movies
n_iter = 100  # Number of epochs
step_size = 0.01  # Learning rate or Step size (gamma)

P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix

def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(R[R > 0]))

train_errors = []
test_errors = []

#Only consider non-zero matrix 
users, items = R.nonzero()      
for iter in range(n_iter):
    for u, i in zip(users, items):
        e = R[u, i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient
        P[:,u] += step_size * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
        Q[:,i] += step_size * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix
    train_rmse = rmse(I,R,Q,P) # Calculate root mean squared error from train dataset
    test_rmse = rmse(I2,T,Q,P) # Calculate root mean squared error from test dataset
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    
plt.plot(range(n_iter), train_errors, marker='o', label='Training Data');
plt.plot(range(n_iter), test_errors, marker='v', label='Test Data');
plt.title('SGD-WR Learning Curve')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show() 

R = pd.DataFrame(R)
R_hat=pd.DataFrame(prediction(P,Q))

ratings = pd.DataFrame(data=R.loc[16,R.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16,R.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']

T = pd.DataFrame(T)
T_hat=pd.DataFrame(prediction(P,Q))
predicted_ratings = pd.DataFrame(data=R_hat.loc[16,R.loc[16,:] == 0])
top_10_reco = predicted_ratings.sort_values(by=16,ascending=False).head(10)