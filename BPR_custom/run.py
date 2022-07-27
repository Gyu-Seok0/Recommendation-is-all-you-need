import time
from copy import deepcopy

import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from evaluation import evaluation
from tqdm import tqdm

def run(args, model, gpu, optimizer, train_loader, test_dataset, model_save_path):

    max_epoch, early_stop, es_epoch = args.max_epoch, args.early_stop, args.es_epoch

    save = True if model_save_path is not None else False
    template = {"best_score" : -np.inf, "best_result": -1, "final_result" : -1}
    eval_dict = {5 : deepcopy(template),
                 10 : deepcopy(template),
                 20 : deepcopy(template),
                 'early_stop' : 0,
                 'early_stop_max' : early_stop,
                 'final_epoch' : 0}
    
    model.train()
    for epoch in range(max_epoch):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        print(f"[Epoch:{epoch:04d}] Training Start")
        for batch_user, batch_pos_item, batch_neg_item in tqdm(train_loader):
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)

            #Forward
            output = model(batch_user, batch_pos_item, batch_neg_item)
            batch_loss = model.get_loss(output)
            epoch_loss.append(batch_loss)

            #Backward and Optimizer
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        epoch_loss = float(torch.mean(torch.stack(epoch_loss, dim = 0)))
        toc1 = time.time()

        # evaluation (내일하자.)
        if epoch < es_epoch:
            verbose = 25
        else:
            verbose = 1
        
        if epoch % verbose == 0:
            is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
            print(f"[eval_results]:{eval_results}")

            if is_improved:
                if save:
                    torch.save(model.state_dict(), model_save_path)
            
        if (eval_dict["early_stop"] >= eval_dict["early_stop_max"]):
            print("Early Stopping!")
            break
    
    print("Last Result", eval_dict)
