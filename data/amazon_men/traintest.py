import pandas as pd

data = pd.read_csv('../data/amazon_men/pos.txt', sep='\t', header=None)
data.columns = ['user', 'item']

train = {}
for index, line in data.iterrows():
    u = int(line['user'])
    i = int(line['item'])
    if train.get(u) is None:
        train[u] = []
    train[u].append(i)

test = {}
for user in train.keys():
    test[user] = train[user].pop()

print('Start Store')
train_dataset = pd.DataFrame()
for user in train.keys():
    for item in train[user]:
        train_dataset = train_dataset.append({'user': user, 'item': item}, ignore_index=True)

test_dataset = pd.DataFrame()
for user in test.keys():
    test_dataset = test_dataset.append({'user': user, 'item': test[user]}, ignore_index=True)

train_dataset = train_dataset.sort_values(by=['user'])
train_dataset[['user', 'item']].to_csv('train.txt', sep='\t', header=None, index=None)
test_dataset[['user', 'item']].to_csv('test.txt', sep='\t', header=None, index=None)

# TODO

# Per analizzare bene le racommandazioni ho bisogno di dividere bene train a test tale  che gli item sono sovrapponibilit. Probabilmente ho bisognod i una limite sul numero di item (cold items) messo almeno a 5

print('End')