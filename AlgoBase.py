import pandas as pd 
import numpy as np
import time
from sklearn.metrics import mean_squared_error, pairwise
from scipy import stats


class AlgoBase(object):
    def __init__(self, learning_rate=.005, regularization=0.02, n_epochs=20,
                 n_factors=100, min_rating=1, max_rating=5):
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.list_val_rmse = []
        self.list_train_rmse = []
        self.n_items = 0
        self.n_users = 0
        self.n_ratings = 0


    def _initialization(self):
        #to be implemented in child class 
        pass 
    

    def _preprocess_data(self, X, train=True):
        """Maps users and items ids to indexes and returns a numpy array.
        Args:
            X (pandas DataFrame): dataset.
            train (boolean): whether or not X is the training set or the
                validation set.
        Returns:
            X (numpy array): mapped dataset.
        """
        X = X.copy()

        if train: # only create u_id, i_id dictionary when preprocessing training data
            u_ids = X['u_id'].unique().tolist()
            i_ids = X['i_id'].unique().tolist()
            self.n_users = len(u_ids)
            self.n_items = len(i_ids)
            self.n_ratings = X.shape[0]
            self.user_dict = dict(zip(u_ids, [i for i in range(self.n_users)]))
            self.item_dict = dict(zip(i_ids, [i for i in range(self.n_items)]))

        X['u_id'] = X['u_id'].map(self.user_dict)
        X['i_id'] = X['i_id'].map(self.item_dict)

        # Tag unknown users/items with -1 (validation data may have unseen user/items)
        X.fillna(-1, inplace=True)

        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)

        X = X[['u_id', 'i_id', 'rating']].values
        return X



    def _create_indicator_matrix(self):
        '''create the indicator matrix from training data
        , where I[user,item] indicate whether user rated item 
        '''
        
        self.I = np.zeros((self.n_users, self.n_items))
        for u,i in self.train[:,:2]:
            self.I[int(u), int(i)] = 1
        self.n_user_rated = np.sum(self.I, axis = 1)
        self.n_item_rated = np.sum(self.I, axis = 0)


    def _prepare_user_inv_kernel_matrix(self, user_kernel_fn):
        user_friends_df = pd.read_csv('user_friends.dat',encoding="utf-8",sep="\t")
        # select the entries only involving users in our training set. 
        user_friends_df = user_friends_df.loc[user_friends_df['userID'].isin(self.user_dict) \
                            & user_friends_df['friendID'].isin(self.user_dict)]
        user_friends_df['userID'] = user_friends_df['userID'].map(self.user_dict)
        user_friends_df['friendID'] = user_friends_df['friendID'].map(self.user_dict)
        user_edge_list = user_friends_df.values.astype(np.int32)

        self.Su = user_kernel_fn(user_edge_list, self.n_users)



    def _prepare_item_inv_kernel_matrix(self, item_kernel_fn):
        user_taggedartists_df= pd.read_csv('user_taggedartists.dat',encoding="utf-8",sep="\t")
        artist_tag_df = user_taggedartists_df.loc[user_taggedartists_df.artistID.isin(self.item_dict)][['artistID','tagID']]
        #map tag id's to indices, map artist id's to indices
        tag_ids = artist_tag_df.tagID.unique().tolist()
        self.n_tags = len(tag_ids)
        self.tag_dict = dict(zip(tag_ids, [i for i in range(self.n_tags)]))
        artist_tag_df.artistID = artist_tag_df.artistID.map(self.item_dict)
        artist_tag_df.tagID = artist_tag_df.tagID.map(self.tag_dict)
        # create artist tag matrix 
        artist_tag_matrix = np.zeros((self.n_items, self.n_tags))
        for a, t in artist_tag_df.values.astype(np.int32):
            artist_tag_matrix[a-1 , t-1] = 1 
        # create the inverse covariance matrix 
        self.Sv = item_kernel_fn(artist_tag_matrix)

    def _on_epoch_begin(self, epoch_ix):
        """Displays epoch starting log and returns its starting time.
        Args:
            epoch_ix: integer, epoch index.
        Returns:
            start (float): starting time of the current epoch.
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs), end=end)

        return start

    def _on_epoch_end(self, start, train_rmse= None, val_rmse=None):
        """
        Displays epoch ending log. If self.verbose compute and display
        validation metrics (loss/rmse/mae).
        # Arguments
            start (float): starting time of the current epoch.
            train_rmse: float, training rmse
            val_rmse: float, validation rmse
        """
        end = time.time()

        print('train_rmse: {:.3f}'.format(train_rmse), end=' - ')
        print('val_rmse: {:.3f}'.format(val_rmse), end=' - ')

        print('took {:.1f} sec'.format(end - start))

    def _sgd(self):
        """Performs SGD algorithm on training data, learns model parameters.
        if not using early stopping, record all validation error and train error in a list 
        if using early stopping, dont modify list_val_rmse and list_train_rmse
        stop after validation error is larger than the min validation error reached. 
        """
        self._initialization()
        #reset the lists
        self.list_val_rmse = [] #reset the lists 
        self.list_train_rmse = []
        self.min_val = 999
            
            

        # Run SGD
        for epoch_ix in range(self.n_epochs):
            start_time = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                np.random.shuffle(self.train)

            self._run_epoch()
            val_rmse = self._compute_metrics(self.val)
            train_rmse = self._compute_metrics(self.train)
            
            self.list_val_rmse.append(val_rmse)
            self.list_train_rmse.append(train_rmse)
            self.min_val = min(val_rmse, self.min_val)                

            self._on_epoch_end(start_time, train_rmse, val_rmse)

            # if early stopping and validation rmse didn't reduce enough , then break 
            if self.early_stopping and self.list_val_rmse[-1] - self.min_val > 0.01:
                break



    def fit(self, train = None, val = None, early_stopping=False, shuffle=True, \
        user_side = False, user_kernel_fn = None, item_side = False, item_kernel_fn = None):
        #Learns model parameters.always require validation data 
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.user_side = user_side
        self.item_side = item_side
        print('Preprocessing data...')
        self.train = self._preprocess_data(train)
        self.val = self._preprocess_data(val, train=False)
        self._create_indicator_matrix()
        self.global_mean = np.mean(self.train[:, 2])
        self.Su = np.diag(np.ones(self.n_users))
        self.Sv = np.diag(np.ones(self.n_items))
        if user_side: 
            print ('Preparing user side information')
            #inverse of kernel matrix 
            self._prepare_user_inv_kernel_matrix(user_kernel_fn)
        if item_side: 
            print ('Preparing item side information')
            #inverse of kernel matrix 
            self._prepare_item_inv_kernel_matrix(item_kernel_fn)

        self._sgd()

        return self




    def predict(self, X):
        #Returns estimated ratings of several given user/item pairs.
        predictions = []

        for u_id, i_id in zip(X['u_id'], X['i_id']):
            predictions.append(self.predict_pair(u_id, i_id))

        return predictions