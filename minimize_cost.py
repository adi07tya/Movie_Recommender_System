import pandas as pd
import numpy as np
from scipy import optimize
import scipy
from utility import *
import warnings
warnings.filterwarnings("ignore")

np.random.seed(142)
data_shape = (943, 1682)
reg_param = 30
ratings = pd.read_csv('data/dataset.base', sep='\t', encoding='latin-1', names=['user_id','movie_id','rating','timestamp']).drop('timestamp',axis=1)
values = ratings.values
values[:, 0:2] -= 1

ratings = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=np.float, shape=data_shape).T
ratings = np.array(ratings.toarray())
num_users = ratings.shape[1]
num_movies = ratings.shape[0]
did_rate = (ratings != 0) * 1
ratings, ratings_mean = normalize_ratings(ratings, did_rate)

num_features = 5
movie_features = np.random.randn( num_movies, num_features )
user_prefs = np.random.randn( num_users, num_features )
initial_X_and_theta = np.r_[movie_features.T.flatten(), user_prefs.T.flatten()]

minimized_cost_and_optimal_params = optimize.fmin_cg(cost, fprime=gradient, x0=initial_X_and_theta,
                                                     args=(ratings, did_rate, num_users, num_movies, num_features, reg_param),
                                                     maxiter=10000, disp=True, full_output=True )
all_prediction = movie_features.dot(user_prefs.T)
print("************************************")
print("Predicted value for user 1")
print("************************************")
print(all_prediction[:,0:1] + ratings_mean)
print("************************************")

print("Original value for user 1")
print("************************************")
print(ratings[:,0:1] + ratings_mean)
print("************************************")
