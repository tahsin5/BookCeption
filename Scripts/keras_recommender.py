import pandas as pd
import numpy as np
import seaborn as sns
import time
import datetime
import random
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('../data/ratings.csv')
books = pd.read_csv('../data/books.csv')

train, test = train_test_split(dataset, test_size=0.2, random_state=42)
n_users = len(dataset.user_id.unique())
n_books = len(dataset.book_id.unique())

from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model

start = time.time()

book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

import keras.backend as K
def rmse (y_true, y_pred):
   return K.sqrt(K.mean(K.square(y_pred -y_true), axis=0))

prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])
model = Model([user_input, book_input], prod)
model.compile('adam', 'mean_squared_error', metrics=['mae',rmse])

history = model.fit([train.user_id, train.book_id], train.rating, batch_size=200, epochs=20, verbose=1)
model.save('regression_model.h5')
cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

# Extract embeddings
book_em = model.get_layer('Book-Embedding')
book_em_weights = book_em.get_weights()[0]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(book_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])

## Creating dataset for making recommendations for the first user
book_data = np.array(list(set(dataset.book_id)))
user = np.array([1 for i in range(len(book_data))])
predictions = model.predict([user, book_data])
predictions = np.array([a[0] for a in predictions])
recommended_book_ids = (-predictions).argsort()[:5]
print(recommended_book_ids)
print(predictions[recommended_book_ids])

# recommended_book = []
# recommended_book.append(books.loc[9051,:][9])
# recommended_book.append(books.loc[8372,:][9])
# recommended_book.append(books.loc[9576,:][9])
# recommended_book.append(books.loc[9089,:][9])
# recommended_book.append(books.loc[9841,:][9])

print(*recommended_book, sep='\n')
#
#
##%%
#pd.reset_option('display.max_columns')
#print(books[books['id'].isin(recommended_book_ids)])

