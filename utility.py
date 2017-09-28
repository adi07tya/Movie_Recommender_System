import numpy as np
def parameters(X_and_theta, num_users, num_movies, num_features):
	first_parameter = X_and_theta[:num_movies * num_features]
	mat1 = first_parameter.reshape((num_features, num_movies)).T
	second_parameter = X_and_theta[num_movies * num_features:]
	mat2 = second_parameter.reshape(num_features, num_users )
	return mat1,mat2

def gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = parameters(X_and_theta, num_users, num_movies, num_features)
	difference = X.dot(theta) * did_rate - ratings
	X_grad = difference.dot( theta.T ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta.T
	return np.r_[X_grad.T.flatten(), theta_grad.T.flatten()]

def cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = parameters(X_and_theta, num_users, num_movies, num_features)
	cost = np.sum( (X.dot( theta ) * did_rate - ratings) ** 2 ) / 2
	regularization = (reg_param / 2) * (np.sum( theta.T**2 ) + np.sum(X**2))
	return cost + regularization

def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    ratings_mean = np.zeros((num_movies, 1))
    ratings_norm = np.zeros(ratings.shape)
    for i in range(num_movies): 
        idx = np.where(did_rate[i] == 1)[0]
        ratings_mean[i] = np.mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    return ratings_norm, ratings_mean
