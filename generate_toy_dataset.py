import pandas as pd 
import numpy as np 
import pickle

user_artists_df = pd.read_csv('user_artists.dat', encoding="utf-8",sep="\t")


#select 1000 most frequent rated artists  (listened to by the largest number of people)
artistID, rate_freq = np.unique(user_artists_df.artistID, return_counts=True)
artist_rate_freq_df = pd.DataFrame({'artistID':artistID, 'rate_freq':rate_freq})
artist_rate_freq_df.sort_values('rate_freq', ascending=False, inplace=True)
top_artists = artist_rate_freq_df.artistID.iloc[:1000].values

toy_df = user_artists_df.loc[user_artists_df['artistID'].isin(top_artists)]

#toy_df = pd.read_csv("toy_data.csv")
#switch columns
data = toy_df[['userID', 'artistID', 'log_weight','weight']]
# rename columns 
data = data.rename(columns={"userID": "u_id", "artistID": "i_id", "log_weight": "rating"})

data_dict = {}

#prepare train/val/test dataset
data_dict['train_80'] = data.sample(frac=0.8, random_state=1)
data_dict['train_20'] = data_dict['train_80'].sample(frac = 1/4, random_state=3)
data_dict['val'] = data.drop(data_dict['train_80'].index.tolist()).sample(frac=0.5, random_state=2)
data_dict['test'] = data.drop(data_dict['train_80'].index.tolist()).drop(data_dict['val'].index.tolist())
print ('Saving data dictionary to local.')
with open('data_dict.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def print_data_descriptions(data):
    num_ratings = data.shape[0]
    num_users = len(data.userID.unique())
    num_artists = len(data.artistID.unique())
                    
    print('we have',num_ratings, 'ratings')
    print('the number of unique users we have is:', num_users)
    print('the number of unique artists we have is:', num_artists)
    print("The median user rated %d artists."%data.userID.value_counts().median())
    print('The max rating is: %d'%data.weight.max(),"the min rating is: %d"%data.weight.min())
    #rating density
    print("rating density is %f"%(num_ratings/num_users/num_artists))
    
    # rating value distribution 
    data.head()


print_data_descriptions(toy_df)