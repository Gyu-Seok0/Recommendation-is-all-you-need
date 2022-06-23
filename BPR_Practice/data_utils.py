import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset
import config

def load_all(test_num= 100):
	train_data = pd.read_csv(config.train_rating,
							 sep = "\t",
							 header = None,
							 names = ["user","item"],
							 usecols= [0,1],
							dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data["item"].max() + 1
	train_data = train_data.values.tolist()

	# make implicit dataset
	train_mat = sp.dok_matrix((user_num,item_num), dtype = np.float32)
	for x in train_data:
		train_mat[x[0],x[1]] = 1.0
	
	test_data = []
	with open(config.test_negative, "r") as fd:
		line = fd.readline()
		while line != None and line != "":
			arr = line.split('\t')
			u,p_i = eval(arr[0])
			test_data.append([u, p_i]) # positive
			for n_i in arr[1:]:
				test_data.append([u,int(n_i)])
			line = fd.readline()

	return train_data, test_data, user_num, item_num, train_mat

class BPRData(Dataset):
	def __init__(self, features, num_item, train_mat = None, num_ng = 0, is_training = None):
		super(BPRData,self).__init__()
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training

		self.features_fill = []
		for x in self.features:
			u, i = x
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u,j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u,i,j])
		

	def __len__(self):
		return self.num_ng * len(self.features) if self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training else self.features
		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if self.is_training else features[idx][1]
		return user,item_i,item_j


