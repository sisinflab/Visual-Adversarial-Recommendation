import pandas as pd
import os
import write

dataset = 'tradesy'
ratings = pd.read_csv('{0}/dataset_preprocessing/{1}/filtered_ratings.txt'.format(os.getcwd(), dataset), sep='\t')
                      # , header=None)
# try:
#     ratings.columns = ['item', 'user', 'rating']
# except:
#     ratings.columns = ['item', 'user', 'rating', 't']

if dataset in ['amazon_sport']:
    ratings.columns = ['item', 'user', 'rating']
else:
    ratings.columns = ['user', 'item', 'rating']

counts = ratings.groupby(['user'])['user'].agg('count').to_frame('count').reset_index()

# Filter rating value to have implicit
# ratings = ratings[ratings['rating'] > 3.0]

core = 10
print('******* CORE: {0} *******'.format(core))
# Filter by 5 ratings
ratings = ratings[ratings['user'].isin(counts[counts['count'] >= core]['user'])]

counts = ratings.groupby(['item'])['item'].agg('count').to_frame('count').reset_index()
ratings = ratings[ratings['item'].isin(counts[counts['count'] >= core]['item'])]

# ratings = ratings.sort_values(['timestamp'])

# Indexing
n_users = ratings['user'].nunique()
n_items = ratings['item'].nunique()

print('NUM USERS: {0}\nNUM ITEMS: {1}\nNUM RATINGS:{2}'.format(n_users, n_items, len(ratings)))

users_index = dict(zip(sorted(ratings['user'].unique()), range(0, n_users)))

numpy_data = []
items_index = dict()
items_to_print = pd.DataFrame()
print('Creating items_ids.csv')
for index, file in enumerate(ratings['item'].unique()):
    items_index[file] = index
    items_to_print = items_to_print.append({'item': file, 'id': int(index)}, ignore_index=True)

items_to_print['id'] = items_to_print['id'].astype(int)
items_to_print.to_csv('{0}/dataset_preprocessing/{1}/items_ids.csv'.format(os.getcwd(), dataset), index=None,
                      header=None)
print('items_ids.csv Created')

del items_to_print
del numpy_data
print('Create var with all f_resnet')

ratings['user'] = ratings['user'].map(users_index)
ratings['item'] = ratings['item'].map(items_index)

# Pos file
# ratings[['user', 'item']].to_csv('pos.txt', sep='\t', header=None, index=None)

items = ratings['item'].unique()
users = ratings['user'].unique()

train = {}
for index, line in ratings.iterrows():
    u = int(line['user'])
    i = int(line['item'])
    if train.get(u) is None:
        train[u] = []
    train[u].append(i)

test = {}
for user in train.keys():
    if len(train[user]) > 1:
        test[user] = train[user].pop()

print('Start Store')
train_dataset = pd.DataFrame()
for user in train.keys():
    for item in train[user]:
        train_dataset = train_dataset.append({'user': user, 'item': item}, ignore_index=True)

test_dataset = pd.DataFrame()
for user in test.keys():
    test_dataset = test_dataset.append({'user': user, 'item': test[user]}, ignore_index=True)

train_dataset['user'] = train_dataset['user'].astype(dtype=int)
train_dataset['item'] = train_dataset['item'].astype(dtype=int)
train_dataset['rating'] = 1.0
train_dataset['timestamp'] = 0
test_dataset['user'] = test_dataset['user'].astype(dtype=int)
test_dataset['item'] = test_dataset['item'].astype(dtype=int)
test_dataset['rating'] = 1.0
test_dataset['timestamp'] = 0

test_dataset_items_only_in_test = test_dataset[~test_dataset['item'].isin(train_dataset['item'].unique())]
# We should drop from test and move in train

for moving_item in test_dataset_items_only_in_test['item'].unique():
    row = test_dataset_items_only_in_test[test_dataset_items_only_in_test['item'] == moving_item].sample()
    test_dataset = test_dataset[
        (test_dataset['user'] != row['user'].values[0]) & (test_dataset['item'] != row['item'].values[0])]
    train_dataset = train_dataset.append({'user': row['user'].values[0], 'item': row['item'].values[0]},
                                         ignore_index=True)

train_dataset = train_dataset.sort_values(by=['user'])
train_dataset[['user', 'item', 'rating', 'timestamp']].to_csv('{0}/dataset_preprocessing/{1}/trainingset.tsv'.format(os.getcwd(), dataset),
                                       sep='\t', header=None, index=None)
test_dataset = test_dataset.sort_values(by=['user'])
test_dataset[['user', 'item', 'rating', 'timestamp']].to_csv('{0}/dataset_preprocessing/{1}/testset.tsv'.format(os.getcwd(), dataset),
                                      sep='\t', header=None, index=None)

write.save_obj(items_index, '{0}/dataset_preprocessing/{1}/item_indices'.format(os.getcwd(), dataset))
write.save_obj(users_index, '{0}/dataset_preprocessing/{1}/user_indices'.format(os.getcwd(), dataset))

print('END Elaboration')

