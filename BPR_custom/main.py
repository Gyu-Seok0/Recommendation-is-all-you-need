import os
import random
import pickle
import argparse
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from dataset import implicit_CF_dataset, implicit_CF_dataset_test

from Models.BPR import BPR # 앞의 BPR은 파일명, 뒤의 BPR은 Class명으로 구분 (파일과 클래스의 이름이 같으면 이렇게 명시적으로 적어줘야 함)
from config import create_argument
from run import run


def main(args):

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # set the seed for initalization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    print(f"[Done] Setting the seed for initalization")


    with open(args.data, "rb") as f:
        dataset = pickle.load(f)
        user_count, item_count = dataset["user_size"], dataset["item_size"]
        candidate_dict, valid_dict, test_dict = dataset["candidate_dict"], dataset["valid_dict"], dataset["test_dict"]
        all_pair, train_pair = dataset["all_pair"], dataset["train_pair"]
    print(f"[Done] Load the pickle: user_count = {user_count}, item_count = {item_count}")

    train_dataset = implicit_CF_dataset(user_count, item_count, train_pair, all_pair, num_ns = args.num_ns)
    test_dataset = implicit_CF_dataset_test(user_count, valid_dict, test_dict, candidate_dict, args.batch_size)
    print("[Done] Transform the dataset into implicit_CF_dataset")

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle = True, drop_last = True)

    # Model 
    if args.model == "BPR":
        model = BPR(user_count, item_count, args.dim, device)
    else:
        raise ImportError

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)
    print("[Done] Set the Model and Optimizer")

    # Create dataset, model, optimizer
    run(args, model, device, optimizer, train_loader, test_dataset, model_save_path = None)




    



if __name__ == "__main__":
    args = create_argument()
    print(f"args = {args}")

    main(args)