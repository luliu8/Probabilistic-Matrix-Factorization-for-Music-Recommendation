import numpy as np
import time
from sklearn.metrics import mean_squared_error
from AlgoBase import AlgoBase


class KPMF(AlgoBase):
	# Kernelized Probabilistic Matrix Factorization as introduced in [Zhou12]
	# Simplify to Probabilistic Matrix Factorization when kernel matrices are both diagonal 

	def _initialization(self):
		"""
		Initializes biases and latent factor matrixes.
		self.p (numpy array): users latent factor matrix.
		self.q (numpy array): items latent factor matrix.
		"""
		print ("initalizing parameters for KPMF")
		self.p = np.random.normal(0, .1, (self.n_users, self.n_factors))
		self.q = np.random.normal(0, .1, (self.n_items, self.n_factors))



	def _run_epoch(self):
		"""Runs an epoch on training data, updating model parameters (p, q).
		"""
		for i in range(self.n_ratings):
			user, item, rating = self.train[i,:3].astype(np.int32)

			# Predict current rating
			pred = np.dot(self.p[user], self.q[item])


			err = rating - pred


			# Update latent factors

			pu = self.p[user]	

			self.p[user] += self.lr * (err * self.q[item] - \
					self.reg / self.n_user_rated[user] / 2 * (self.Su[user] @ self.p + self.p[user]))

			self.q[item] += self.lr * (err * pu - \
					self.reg / self.n_item_rated[item] / 2 * (self.Sv[item] @ self.q + self.q[item]))


	def _compute_metrics(self, X):
		"""Computes rmse with current model parameters.
		Args: 
		X (numpy array) 
		"""

		residuals = []

		for i in range(X.shape[0]):
			user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
			# predict global mean if user or item is new
			pred = self.global_mean

			if (user > -1) and (item > -1):
				pred = np.dot(self.p[user], self.q[item])
				residuals.append(rating - pred)

		residuals = np.array(residuals)
		loss = np.square(residuals).mean()
		rmse = np.sqrt(loss)

		return rmse




	def predict_pair(self, u_id, i_id, clip=False):
		"""Returns the model rating prediction for a given user/item pair.
		convert u_id, i_id to u_ix and i_ix 
		Args:
		u_id (int): a user id.
		i_id (int): an item id.
		clip (boolean, default is `True`): whether to clip the prediction
		or not.

		Returns:
		pred (float): the estimated rating for the given user/item pair.
		"""
		user_known, item_known = False, False
		pred = self.global_mean

		if u_id in self.user_dict:
			user_known = True
			u_ix = self.user_dict[u_id]

		if i_id in self.item_dict:
			item_known = True
			i_ix = self.item_dict[i_id]

		if  user_known and item_known:
			pred = np.dot(self.p[u_ix], self.q[i_ix])

		if clip:
			pred = self.max_rating if pred > self.max_rating else pred
			pred = self.min_rating if pred < self.min_rating else pred

		return pred
	  
