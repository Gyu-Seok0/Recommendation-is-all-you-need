import numpy as np
import torch

def hit(gt_item, pred_items):
    return 1 if gt_item in pred_items else 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return 1/(np.log2(index+2)) # implicit data이므로 분자에는 1이 들어간다.
    return 0

def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item_i, item_j in test_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        prediction_i, _ = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(item_i, indices).cpu().numpy().tolist()
        gt_item = item_i[0].item()

        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)