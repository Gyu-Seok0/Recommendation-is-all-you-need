import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import time 

#from Utils.data_utils import *
#################################################################################################################
# For training
#################################################################################################################

class implicit_CF_dataset(data.Dataset):
    def __init__(self, user_count, item_count, train_pair, all_pair, num_ns):
        super(implicit_CF_dataset,self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.num_ns = num_ns
        self.all_pair = all_pair
        self.train_pair = train_pair
        self.train_arr = []

    def negative_sampling(self):
        start = time.time()
        for user, pos_item in self.train_pair:
            ns_count = 0
            while True:
                neg_item = np.random.randint(self.item_count)
                if not (user, neg_item) in self.all_pair:
                    self.train_arr.append((user, pos_item, neg_item))
                    ns_count += 1
                    if ns_count >= self.num_ns:
                        break
        end = time.time()
        print(f"Time for Negative Sampling:{end - start:.4f}")

               
    def __len__(self):
        return len(self.train_pair) * self.num_ns
        

    def __getitem__(self, idx):
        assert len(self.train_arr) > 0
        return self.train_arr[idx][0], self.train_arr[idx][1], self.train_arr[idx][2]

#################################################################################################################
# For test
#################################################################################################################

class implicit_CF_dataset_test(data.Dataset):
    def __init__(self, user_count, valid_dict, test_dict, candidates, batch_size):
        super(implicit_CF_dataset_test, self).__init__()

        self.test_item = []
        self.valid_item = []
        self.candidates = []

        num_candidates = len(candidates[0])
        

        for user in range(user_count):
            if user not in test_dict:
                self.test_item.append([0])
                self.valid_item.append([0])
                self.candidates.append([0] * num_candidates)
            else:
                self.test_item.append(int(test_dict[user]))
                self.valid_item.append(int(valid_dict[user]))
                self.candidates.append(candidates[user])

        self.test_item = torch.LongTensor(self.test_item)
        self.valid_item = torch.LongTensor(self.valid_item)
        self.candidates = torch.LongTensor(self.candidates)

        self.user_list = torch.LongTensor(list(test_dict.keys()))

        self.batch_start = 0
        self.batch_size = batch_size        

    def __len__(self):
        return len(self.test_item)

    def get_next_batch_users(self):
        batch_start = self.batch_start
        batch_end = self.batch_start + self.batch_size

        if batch_end >= len(self.user_list):
            batch_end = len(self.user_list)
            self.batch_start = 0
            is_last_batch = True
        else:
            self.batch_start += self.batch_size
            is_last_batch = False
        return self.user_list[batch_start:batch_end], is_last_batch
    
    def get_next_batch(self, batch_user):
        batch_test_items = torch.index_select(self.test_item, dim = 0, index = batch_user)
        batch_valid_items = torch.index_select(self.valid_item, dim = 0, index = batch_user)
        batch_candidates = torch.index_select(self.candidates, dim = 0, index = batch_user)

        return batch_test_items, batch_valid_items, batch_candidates