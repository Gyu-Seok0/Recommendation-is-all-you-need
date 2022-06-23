'''
전체적인 개요
1) 데이터 전처리
2) 간단한 MF모델링
3) Train/Test set만들기
4) 학습시키기(이때 BPR loss를 구현할 것)
5) TEST 시키기
+ Evaluation Metric으로 HR, NDCG 사용하면 될듯.
'''

import os
import time
import argparse
import numpy as np

import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import data_utils
from model import BPR
from evaluate import metrics
from log import get_logger
import config


def main(args):
    logger = get_logger()

    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()
    train_dataset = data_utils.BPRData(features = train_data,
                                        num_item = item_num,
                                        train_mat = train_mat,
                                        num_ng = args.num_ng,
                                        is_training = True)

    test_dataset = data_utils.BPRData(features = test_data,
                                        num_item = item_num,
                                        train_mat = train_mat,
                                        num_ng = 0,
                                        is_training = False)

    train_loader = DataLoader(train_dataset,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers = 4)
    
    test_loader = DataLoader(test_dataset,
                             batch_size = args.test_num_ng + 1,
                             shuffle = False,
                             num_workers = 0)
    
    # modeling
    model = BPR(user_num = user_num,
                item_num = item_num,
                factor_num = args.factor_num)
    model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr = args.lr,
                          weight_decay = args.lamda)
    
    HR, NDCG = metrics(model, test_loader, args.top_k)

    # training
    count, best_hr = 0, 0
    for epoch in range(args.epochs):
        logger.info(f"[Epoch:{epoch:04d}] Training Start!")
        
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item_i, item_j in train_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()

            model.zero_grad()
            prediction_i, prediction_j = model(user,item_i,item_j)
            loss = - (prediction_i - prediction_j).sigmoid().log().sum()
           
            
            # backward
            loss.backward()
            optimizer.step()
            count += 1

        model.eval()
        HR, NDCG = metrics(model, test_loader, args.top_k)
        elapsed_time = time.time() - start_time

        logger.info("Training Done! it tooks time:")
        logger.info(time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                torch.save({"model_state": model.state_dict(),
                            "epoch":epoch,
                            "hr":best_hr,
                            "ndcg":best_ndcg},config.model_path + f"epoch_{epoch}_hr_{HR}_ndcg_{NDCG}.tar")
        logger.info(f"Result of Test: HR = {HR:.4f}, NDCG = {NDCG:.4f}")

    # End
    logger.info(f"End. Best epoch:{best_epoch:03d}: HR = {best_hr:.4f},NDCG = {best_ndcg:.4f}")
    return 0

if __name__ == "__main__":
    # argument assign
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default=0.01, help = "learning_rate")
    parser.add_argument("--lamda",type = float, default=0.001, help = "regularization rate")
    parser.add_argument("--batch_size",type = int, default=4096, )
    parser.add_argument("--epochs",type = int, default=50, help="training epochs")
    parser.add_argument("--top_k", type = int, default=10, help="compute metrics like @top_k")
    parser.add_argument("--factor_num",type = int, default=32, help = "embedding size")
    parser.add_argument("--num_ng", type = int, default=4,help="sample part of negative items for testing")
    parser.add_argument("--out", default = True, help = "save model or not")
    parser.add_argument("--gpu", type = str, default = "2", help = "gpu card ID") # 이따 멀티 gpu 돌려보기
    parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
    parser.add_argument("--log", default = True, help = "logging or not")

    # argumanet aggregate
    args = parser.parse_args()
    
    # execute
    main(args)