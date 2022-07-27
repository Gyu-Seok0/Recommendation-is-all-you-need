from copyreg import pickle
import os
import pandas as pd
import numpy as np
import argparse
import math
import pickle
import copy


class MovieLens100k(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load(self):
        file_path = os.path.join(self.data_dir, "u.data")
        df = pd.read_csv(file_path,
                         sep = "\t",
                         names = ["user_id","item_id","rating","time"],
                         usecols = ["user_id","item_id","time"])
        return df

def convert_unique_idx(df, column_name):
    column_dict = {x:i for i,x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype(int)
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

def create_user_list(df, user_size):
    user_list = [list() for _ in range(user_size)]
    for row in df.itertuples():
        user_list[row.user_id].append((row.time, row.item_id))
    return user_list

def split_train_test(df, user_size, test_size = 0.2, time_order = False):
    if not time_order: #시간이 고려하지 않는 경우
        test_idx = np.random.choice(len(df), int(len(df)*test_size), replace = False) # 중복방지
        train_idx = list(set(range(len(df))) - set(test_idx))
        
        test_df = df.iloc[test_idx].reset_index(drop = True) # drop = True -> index라는 column을 drop 시킴
        train_df = df.iloc[train_idx].reset_index(drop = True)

        test_user_list = create_user_list(test_df, user_size)
        train_user_list = create_user_list(train_df, user_size)
    else: #시간을 고려
        total_user_list = create_user_list(df, user_size)
        train_user_list = [None] * len(df)
        test_user_list = [None] * len(df)
        for user, item_list in enumerate(total_user_list):
            # sort(오래된거 -> 최근꺼)
            item_list.sort(key = lambda x: x[0])
            train_test_split_idx = math.ceil(len(item_list)*(1-test_size))
            train_user_list[user] = item_list[:train_test_split_idx]
            test_user_list[user] = item_list[train_test_split_idx:]
    
    # Remove Time
    test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]
    train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]
    return train_user_list, test_user_list

def create_leave_one_out(df, user_size, item_size, num_ns = 99):
    total_dict = {user : [] for user in range(user_size)}
    candidate_dict = {user : [] for user in range(user_size)}
    train_dict = {}
    valid_dict = {}
    test_dict = {}

    for row in df.itertuples():
        total_dict[row.user_id].append(row.item_id)

    train_dict = copy.deepcopy(total_dict)
    
    for user, items in train_dict.items():       
        if len(items) > 2:
            valid_item, test_item = np.random.choice(items, size = 2, replace = False)
            valid_dict[user] = valid_item
            test_dict[user] = test_item

            num_neg_sample = 0
            while True:
                neg_sample = np.random.choice(range(item_size))
                if neg_sample not in items:
                    candidate_dict[user].append(neg_sample)
                    num_neg_sample += 1
                    if num_neg_sample >= num_ns:
                        break
            # 삭제
            train_dict[user].remove(valid_item)
            train_dict[user].remove(test_item)

    return total_dict, candidate_dict, train_dict, valid_dict, test_dict    

def create_pair(user_dict):
    pair = []
    for user, item_list in user_dict.items():
        pair.extend([(user,item) for item in item_list])
    return pair


# def create_pair(user_list):
#     pair = []
#     for user, item_list in enumerate(user_list):
#         pair.extend([(user, item) for item in item_list])
#     return pair


def main(args):
    if args.dataset == "ml-100k":
        df = MovieLens100k(args.data_dir).load()
    else:
        raise NotImplementedError

    df, user_mapping = convert_unique_idx(df, "user_id")
    df, item_mapping = convert_unique_idx(df, "item_id")
    print('Complete assigning unique index to user and item')

    user_size = len(df["user_id"].unique())
    item_size = len(df["item_id"].unique())
    print(f"user_size = {user_size}, item_size = {item_size}")

    total_dict, candidate_dict, train_dict, valid_dict, test_dict = create_leave_one_out(df, user_size, item_size)
    all_pair = create_pair(total_dict)
    train_pair = create_pair(train_dict)

    '''
    train_user_list, test_user_list = split_train_test(df,
                                                       user_size,
                                                       test_size = args.test_size,
                                                       time_order = args.time_order)
    '''

    print('Complete spliting items for training and testing') 

    dataset = {"user_size" : user_size,
               "item_size" : item_size,
               "user_mapping" : user_mapping,
               "item_mapping" : item_mapping,
               "all_pair" : all_pair,
               "train_pair" : train_pair,
               "candidate_dict" : candidate_dict,
               "train_dict" : train_dict,
               "valid_dict" : valid_dict,
               "test_dict" : test_dict}
    
    dirname = os.path.dirname(os.path.abspath(args.output_data))
    os.makedirs(dirname, exist_ok = True)
    with open(args.output_data, "wb") as f:
        pickle.dump(dataset, f, protocol = pickle.HIGHEST_PROTOCOL)
        print("here")
    
    print(' [Done] Prepare the Train/Valid/Testset as Leave-One-Out Setting! ') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = "ml-100k")
    parser.add_argument("--data_dir", type = str, default = "../ml-100k")
    parser.add_argument("--test_size", type = float, default = 0.2)
    parser.add_argument("--time_order", type = bool, default = False)
    parser.add_argument("--output_data", type = str, default = os.path.join("preprocessed","cand_ml-100k.pickle"))

    args = parser.parse_args()

    main(args)
