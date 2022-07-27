import copy
import time
import torch
import math
import numpy as np
from tqdm import tqdm 

def to_np(x):
    return x.data.cpu().numpy()

def check(ranking_list, target_item, topk=10):
    k = 0
    for item_id in ranking_list:
        if target_item == item_id:
            return (1., math.log(2.)/ math.log(k + 2.), 1.0/ (k + 1.))
        if k >= topk:
            return (0., 0., 0.)
        k += 1
        

def latent_factor_evaluate(model, test_dataset):
    metrics = {"H5":[],  "M5":[],  "N5":[],
               "H10":[], "M10":[], "N10":[],
               "H20":[], "M20":[], "N20":[]}
    
    eval_results = {"test" : copy.deepcopy(metrics),
                    "valid" : copy.deepcopy(metrics)}
    
    # extract score
    if model.sim_type == "inner product":
        user_emb, item_emb = model.get_embedding()
        score_mat = to_np(-torch.matmul(user_emb,item_emb.T))
    
    # user
    test_user_list = to_np(test_dataset.user_list)
    for test_user in tqdm(test_user_list):
        test_item = [int(test_dataset.test_item[test_user].item())]
        valid_item = [int(test_dataset.valid_item[test_user].item())]
        candidates = to_np(test_dataset.candidates[test_user]).tolist()
     

        total_items = test_item + valid_item + candidates
        score = score_mat[test_user][total_items]

        result = np.argsort(score).flatten().tolist() # 가장 점수가 작은 아이템의 인덱스부터 순서대로 리스트에 담김.
        ranking_list = np.array(total_items)[result]
        # print("score",score)
        # print("result",result)
        # print("ranking_list", ranking_list)
        # print("test_item", test_item)
        # print("valid_item",valid_item)
        
        for mode in ['test','valid']:
            if mode == "test":
                target_item = test_item[0]
                ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == valid_item[0]))
            else:
                target_item = valid_item[0]
                ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == test_item[0]))

            for topk in [5, 10, 20]:
                (h, n, m) = check(ranking_list_tmp, target_item, topk)
                eval_results[mode]["H" + str(topk)].append(h)
                eval_results[mode]["N" + str(topk)].append(n)
                eval_results[mode]["M" + str(topk)].append(m)

    for mode in ["test","valid"]:
        for topk in [5,10,20]:
            eval_results[mode]["H" + str(topk)] = round(np.asarray(eval_results[mode]["H" + str(topk)]).mean(),4)
            eval_results[mode]["N" + str(topk)] = round(np.asarray(eval_results[mode]["N" + str(topk)]).mean(),4)
            eval_results[mode]["M" + str(topk)] = round(np.asarray(eval_results[mode]["M" + str(topk)]).mean(),4)

    return eval_results

def evaluation(model, gpu, eval_dict, epoch, test_dataset):

    print("Evaluation Start")
    model.eval()
    with torch.no_grad():
        tic = time.time()
        if model.sim_type == "inner product":
            eval_results = latent_factor_evaluate(model, test_dataset)
        else:
            raise NotImplementedError
        
 
        toc = time.time()
        is_improved = False

        for topk in [5,10,20]:
            if eval_dict['early_stop'] < eval_dict["early_stop_max"]:
                if eval_dict[topk]["best_score"] < eval_results["valid"]["H" + str(topk)]:
                    eval_dict[topk]['best_score'] = eval_results['valid']['H' + str(topk)]
                    eval_dict[topk]['best_result'] = eval_results['valid']
                    eval_dict[topk]['final_result'] = eval_results['test']
                    
                    is_improved = True
                    eval_dict['final_epoch'] = epoch
        if not is_improved:
            eval_dict['early_stop'] += 1
        else:
            eval_dict["early_stop"] = 0
        
        return is_improved, eval_results, toc - tic
