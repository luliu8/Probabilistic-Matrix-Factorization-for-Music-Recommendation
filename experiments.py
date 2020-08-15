import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, pairwise
import pickle
from KPMF import KPMF
from cKPMF import cKPMF

from my_kernels import *
n_factor = 30

# read data dictionary
with open('data_dict.pickle', 'rb') as handle:
    data_dict = pickle.load(handle)
# read hyperparameters (obtained via gridsearch ) 
with open('KPMF_hyper_{}.pickle'.format(n_factor), 'rb') as handle:
    KPMF_hyper = pickle.load(handle)
with open('cKPMF_hyper_{}.pickle'.format(n_factor), 'rb') as handle:
    cKPMF_hyper = pickle.load(handle)



KPMF_result = defaultdict(dict)
cKPMF_result = defaultdict(dict)
	
def fit_and_pred(model, train_size, cur_dict, user_side, item_side):
	# first train the model without early stopping
	model.fit(train = data_dict[train_size] , val = data_dict['val'], \
		user_side = user_side, user_kernel_fn = inv_commute_time_kernel, \
		item_side = item_side, item_kernel_fn = inv_rbf_kernel)
	cur_dict['list_val_rmse'] = model.list_val_rmse[:]
	cur_dict['list_train_rmse'] = model.list_train_rmse[:]
	
	# fit with early stopping to obtain the best model 
	model.fit(train = data_dict[train_size] , val = data_dict['val'], early_stopping = True, \
		user_side = user_side, user_kernel_fn = inv_commute_time_kernel, \
		item_side = item_side, item_kernel_fn = inv_rbf_kernel)
	cur_dict['test_prediction'] = model.predict(data_dict['test'])
	cur_dict['test_rmse'] = np.sqrt(mean_squared_error(data_dict['test']['rating'], cur_dict['test_prediction']))
	print ("Test RMSE: {:.3f}".format(cur_dict['test_rmse']))

# for each train data and each num_factor: 
# first train all epochs , obtain list_val_rmse and list_train_rmse
# then train with early stopping to obtain the best model 
# calculate test rmse 
fixed_params = {'n_factors': n_factor , 'n_epochs': 100}

for train_size in ['train_80','train_20']:
	# experiments for KPMF model 
	for (user_side, item_side) in [(False, False),(False, True), [True, False], [True, True]]:
		model_key = (train_size, user_side, item_side)
		params = fixed_params.copy()
		params['learning_rate'] = KPMF_hyper[model_key]['lr']
		params['regularization'] = KPMF_hyper[model_key]['reg']

		model = KPMF(**params)
		print ("training KPMF on {} with user_info = {}, item_info ={}"\
			.format(train_size, user_side, item_side))
		cur_dict = KPMF_result[model_key]
		fit_and_pred(model, train_size, cur_dict, user_side, item_side)
		

	# experiments for cKPMF model 
	for (user_side, item_side) in [(False, False),(False, True)]:
		model_key = (train_size, item_side)
		params = fixed_params.copy()
		params['learning_rate'] = cKPMF_hyper[model_key]['lr']
		params['regularization'] = cKPMF_hyper[model_key]['reg']
		model = cKPMF(**params)
		print ("training cKPMF on {} with item_info ={}".format(train_size, item_side))
		cur_dict = cKPMF_result[model_key]
		fit_and_pred(model, train_size, cur_dict, user_side, item_side)
		


print ("dumping result to local")
with open('KPMF_result_{}.pickle'.format(n_factor), 'wb') as handle:
    pickle.dump(KPMF_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('cKPMF_result_{}.pickle'.format(n_factor), 'wb') as handle:
    pickle.dump(cKPMF_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
