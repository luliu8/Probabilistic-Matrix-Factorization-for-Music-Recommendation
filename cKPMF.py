import numpy as np
import time
from sklearn.metrics import mean_squared_error
from AlgoBase import AlgoBase


class cKPMF(AlgoBase):
	# constrained Kernelized Matrix Factorization model 
	# Simplify to constrained Probabilistic Matrix Factorization model when kernel matrix is diagonal

	def _initialization(self):
		#Initializes latent factor matrixes.
		
		print ("initalizing variables for  cKPMF")
		self.y = np.random.normal(0, .1, (self.n_users, self.n_factors))
		self.w = np.random.normal(0, .1, (self.n_items, self.n_factors))
		self.q = np.random.normal(0, .1, (self.n_items, self.n_factors))

		

	def _run_epoch(self): 
		"""Runs an epoch on training set, updating model parameters (y, q, w).
		"""
		for i in range(self.n_ratings):
			user, item, rating = self.train[i,:3].astype(np.int32)

		   

			pu = self.y[user] + (self.I[user] @ self.w)/ self.n_user_rated[user]

			#predict current rating 
			pred = np.dot(pu, self.q[item])
			err = rating - pred
			
		   

			# Update latent factors
			qi = self.q[item]
			wi = self.w[item]
			self.y[user] += self.lr * (err * self.q[item] - self.reg / self.n_user_rated[user] * self.y[user])
			self.q[item] += self.lr * (err * pu - \
					self.reg / self.n_item_rated[item] / 2 * (self.Sv[item] @ self.q + self.q[item]))
			self.w += self.lr *(err * np.outer(self.I[user], qi) / self.n_user_rated[user]) 
			self.w[item] -= self.lr * self.reg / self.n_item_rated[item] * wi 


	def _compute_metrics(self, X):
		"""Computes rmse from current model parameters.
		   Args: 
				X (numpy array) 
		"""

		residuals = []

		for i in range(X.shape[0]):
			user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
			pred = self.global_mean

			
			if (user > -1) and (item > -1):
				pu = self.y[user] + (self.I[user] @ self.w)/ self.n_user_rated[user]
				# Predict current rating
				pred = np.dot(pu, self.q[item])

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
			pu = self.y[u_ix] + (self.I[u_ix] @ self.w)/ self.n_user_rated[u_ix]
			# Predict current rating
			pred = np.dot(pu, self.q[i_ix])

		if clip:
			pred = self.max_rating if pred > self.max_rating else pred
			pred = self.min_rating if pred < self.min_rating else pred

		return pred  
	  
