# grid search for best learning rate and regularization coefficient for all models 

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, pairwise
import pickle
from KPMF import KPMF
from cKPMF import cKPMF

from my_kernels import *

# read data dictionary
with open('data_dict.pickle', 'rb') as handle:
    data_dict = pickle.load(handle)


KPMF_hyper = defaultdict(dict)
cKPMF_hyper = defaultdict(dict)
n_factor  = 30
def grid_search(Model, train_size, user_side, item_side):
	params = {'n_factors': n_factor, 'n_epochs': 100}
	result = {'min_val': 999, 'lr': None, 'reg': None}
	for lr in [0.01, 0.005]:
		for reg in [0.1, 0.01, 0.001]:
			params['learning_rate'] = lr
			params['regularization'] = reg
			m = Model(**params)
			print ("trying learning rate : {}, regulairzation : {}".format(lr, reg))
			# fit with early stopping to obtain the best model 
			m.fit(train = data_dict[train_size] , val = data_dict['val'], early_stopping = True, \
				user_side = user_side, user_kernel_fn = inv_commute_time_kernel, \
				item_side = item_side, item_kernel_fn = inv_rbf_kernel)
			if m.min_val < result['min_val']: 
				result['min_val'] = m.min_val
				result['lr'] = lr
				result['reg'] = reg
	
	print ('best lr: {}, best reg: {}, min val rmse: {}'.format(result['lr'], result['reg'], result['min_val']))
	return result



for train_size in ['train_20', 'train_80']:
	# grid search for KPMF model 
	for (user_side, item_side) in [(False, False),(False, True), [True, False], [True, True]]:
		print ("training KPMF on {}, with user_info = {}, item_info ={}"\
			.format(train_size, user_side, item_side))
		KPMF_hyper[(train_size, user_side, item_side)] = grid_search(KPMF, train_size, user_side, item_side)
		

	# grid search for cKPMF model 
	for (user_side, item_side) in [(False, False),(False, True)]:
		print ("training cKPMF on {} with item_info ={}"\
			.format(train_size,  item_side))
		
		cKPMF_hyper[(train_size, item_side)] = grid_search(cKPMF, train_size, user_side, item_side)
		


print ("dumping result to local")
with open('KPMF_hyper_{}.pickle'.format(n_factor), 'wb') as handle:
    pickle.dump(KPMF_hyper, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('cKPMF_hyper_{}.pickle'.format(n_factor), 'wb') as handle:
    pickle.dump(cKPMF_hyper, handle, protocol=pickle.HIGHEST_PROTOCOL)
