import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import random
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD
import numpy as np
import pandas as pd
from surprise import CoClustering

ratings = pd.read_csv('../data/ratings.csv')
books_data = pd.read_csv('../data/books.csv')
duplicates = ratings.duplicated()
duplicates = duplicates[duplicates == True]

columnsTitles=["user_id","book_id","rating"]
ratings = ratings.reindex(columns=columnsTitles)


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

algo = CoClustering()
start = time.time()
cross_validate(algo, data, measures=['RMSE'], cv=10, verbose=True)
cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))

pred = algo.predict(7563, 1, r_ui=4, verbose=True)

from collections import defaultdict

def get_top_k(predictions, k):
    '''Return a top_k dicts where keys are user ids and values are lists of
    tuples [(item id, rating estimation) ...].

    Takes in a list of predictions as returned by the test method.
    '''

    # First map the predictions to each user.
    top_k = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_k[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_k.items():
        user_ratings.sort(key=lambda x:x[1], reverse=True)
        top_k[uid] = user_ratings[:k]

    return top_k

testset = data.construct_testset(raw_testset=data.raw_ratings)
predictions = algo.test(testset)

top_k = get_top_k(predictions, 10)

# Print the recommended items
for uid, user_ratings in top_k.items():
    print(uid, [iid for (iid, _) in user_ratings])



# Compute the total number of recommended items.
all_recommended_items = set(iid for (_, user_ratings) in top_k.items() for
                            (iid, _) in user_ratings)

print('Number of recommended items:', len(all_recommended_items), 'over',
      len(top_k), 'users')



#import time
#import datetime
#import random
#
#import numpy as np
#import pandas as pd
#import six
#from tabulate import tabulate
#
#from surprise import Dataset
#from surprise import Reader
#from surprise.model_selection import cross_validate
#from surprise.model_selection import KFold
#from surprise import NormalPredictor
#from surprise import BaselineOnly
#from surprise import KNNBasic
#from surprise import KNNWithMeans
#from surprise import KNNBaseline
#from surprise import SVD
#from surprise import SVDpp
#from surprise import NMF
#from surprise import SlopeOne
#from surprise import CoClustering
#
## The algorithms to cross-validate
#classes = (CoClustering, KNNWithMeans)
#
## ugly dict to map algo names and datasets to their markdown links in the table
#stable = 'http://surprise.readthedocs.io/en/stable/'
##LINK = {'KNNWithMeans': '[{}]({})'.format('Centered k-NN',
##                                          stable +
##                                          'knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans'),
##        'CoClustering': '[{}]({})'.format('Co-Clustering',
##                                          stable +
##                                          'co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering'),
##        }
#
#LINK = {'KNNWithMeans': '[{}]({})'.format('Centered k-NN',
#                                          stable +
#                                          'knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans')                                
#        }
#
#
## Real Ratings
#ratings = pd.read_csv('../data/ratings.csv')
#books_data = pd.read_csv('../data/books.csv')
#
#columnsTitles=["user_id","book_id","rating"]
#ratings = ratings.reindex(columns=columnsTitles)
#
#reader = Reader(rating_scale=(1, 5))
#data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
#
## The algorithms to cross-validate
#classes = (CoClustering, KNNWithMeans)
#
## set RNG
#np.random.seed(0)
#random.seed(0)
#
## set KFold
#kf = KFold(n_splits=10, random_state=0)  # folds will be the same for all algorithms.
#
#table = []
#for klass in classes:
#    start = time.time()
#    out = cross_validate(klass(), data, ['rmse'], kf)
#    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
#    link = LINK[klass.__name__]
#    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
#
#    new_line = [link, mean_rmse, cv_time]
#    print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
#    table.append(new_line)
#
#header = ['RMSE',
#          'Time'
#          ]
#print(tabulate(table, header, tablefmt="pipe"))
#
#
