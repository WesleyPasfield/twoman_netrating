import numpy as np
import pandas as pd

def train_valid_test_splitter(df, train_prop, seed):

	np.random.seed(seed)

	dfnew = df

	train_range = np.random.choice(len(dfnew), int(len(dfnew) * train_prop), replace = False)
	train = dfnew.loc[train_range,:]

	test_valid = dfnew.loc[set(dfnew.index) - set(train_range)]
	test_valid = test_valid.reset_index(drop = True)

	valid_range = np.random.choice(len(test_valid), int(len(test_valid) * 0.5), replace = False)

	valid = test_valid.loc[valid_range,:]
	test = test_valid.loc[set(test_valid.index) - set(valid_range)]

	print(len(train))
	print(len(valid))
	print(len(test))

	train_players = train[['player_left', 'player_right', 'team_year']]
	valid_players = valid[['player_left', 'player_right', 'team_year']]
	test_players = test[['player_left', 'player_right', 'team_year']]

	train.drop(['player_left', 'player_right'], inplace = True, axis = 1)
	valid.drop(['player_left', 'player_right'], inplace = True, axis = 1)
	test.drop(['player_left', 'player_right'], inplace = True, axis = 1)

	return train, train_players, valid, valid_players, test, test_players

def cat_encoder(df, cols, encoding=False):

	if not encoding:
		print('doing this')

		val_types = dict()
		val_length = list()
		for col in cols:
			val_types[col] = df[col].unique()

		encoding = dict()
		for k, v in val_types.items():
			encoding.setdefault(k, 0)
			encoding[k] = {o: i for i, o in enumerate(val_types[k])}

	for col in cols:
		uniques = df[col].unique()
		for unique in uniques:
			if unique not in encoding[col]:
				print(unique)
				encoding[col][unique] = 0
	print('go')

	for k, v in encoding.items():
		df[k] = df[k].apply(lambda x: v[x])

	df = pd.DataFrame(df)

	return df, encoding

def min_max(df, base_df, cols):

	for col in cols:

		min_val = base_df[col].min()
		max_val = base_df[col].max()

		df[col] = np.where((df[col] - min_val) / (max_val - min_val) > 1.0, 1.0,
						   (df[col] - min_val) / (max_val - min_val))

	return df

def xandy(df, variable):

	df = df.reset_index(drop=True)
	yval = df[variable]
	df.drop([variable], inplace = True, axis = 1)

	return df, yval