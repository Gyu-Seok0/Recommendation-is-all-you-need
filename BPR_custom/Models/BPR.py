from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR(nn.Module):
    def __init__(self, user_count, item_count, dim, gpu):
        super(BPR,self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count

        self.user_list = torch.LongTensor([i for i in range(user_count)])
        self.item_list = torch.LongTensor([i for i in range(item_count)])

        if gpu is not None:
            self.user_list = self.user_list.to(gpu)
            self.item_list = self.item_list.to(gpu)

        self.user_emb = nn.Embedding(self.user_count, dim)
        self.item_emb = nn.Embedding(self.item_count, dim)

        # init
        nn.init.normal_(self.user_emb.weight, mean = 0., std = 0.01)
        nn.init.normal_(self.item_emb.weight, mean = 0., std = 0.01)

        # user-item similarity type
        self.sim_type = "inner product"        

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.user_emb(batch_user) 
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)

        pos_score = (u*i).sum(dim = 1, keepdim = True) # Batch * 1
        neg_score = (u*j).sum(dim = 1, keepdim = True)
        output = (pos_score, neg_score)

        return output
    
    def get_loss(self, output):
        pos_score, neg_score = output
        loss = -(pos_score - neg_score).sigmoid().log().sum() # mean이 아니라 sum을 취해버리네
        return loss
    
    def forward_multi_items(self, batch_user, batch_items):

        batch_user = batch_user.unsqueeze(-1) # 1-D -> 2-D
        batch_user = torch.cat(batch_items.size(1) * batch_user, dim = 1)

        u = self.user_emb(batch_user) # batch_size x k x dim
        i = self.item_emb(batch_items)

        score = (u * i).sum(dim = -1, keepdim = False)
        return score

    def get_embedding(self):
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)
        return users, items
