import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import spatial

import time

epsilon = 1e-5


class NeuralMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(NeuralMatrixFactorization, self).__init__()

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.dim = args.dim
        self.train_label = args.train_label
        self.cluster = args.cluster
        self.norm = args.norm
        self.scale1 = args.scale1
        self.scale2 = args.scale2


        self.user_embeddings = nn.Embedding(num_embeddings=self.num_users, embedding_dim=args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(num_embeddings=self.num_items + 1, embedding_dim=args.dim, padding_idx=self.num_items).to(self.device)
        self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)     

        # MLP part
        #self.transform = nn.Sequential(
        #    nn.Linear(self.dim*2, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, self.dim),
        #    nn.ReLU()).to(self.device)  


        self.transform = nn.Sequential(
            nn.Linear(self.dim*3, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)).to(self.device)  

        # concat the MF part and MLP part
        self.combine = nn.Sequential(
            nn.Linear(self.dim * 2, 1)).to(self.device) 


    def inference(self, users_embed, items_embed):
        self.mf = torch.mul(users_embed, items_embed) # MF part
        return self.transform(torch.cat((users_embed, items_embed, self.mf), 1))
        #self.mlp = self.transform(torch.cat((users_embed, items_embed), 1)) # MLP part
        #self.prediction = self.combine(torch.cat((self.mf, self.mlp), 1))
        #return self.prediction


    def community_ui(self, user_id):
        batch_label = self.train_label[user_id]
        # if self.thres == 'N':
        pos_i_com = torch.matmul(batch_label, self.item_embeddings.weight)
        num = batch_label.sum(1).unsqueeze(1)
        pos_i_com = pos_i_com / num
        # elif self.thres == 'Y':
        #     batch_u_emb = self.agg_user_embeddings(user_id)
        #     batch_sim = self.compute_cosine(batch_u_emb, self.agg_item_embeddings.weight)
        #     pos_i_com = self.neighbor(batch_label, batch_sim, -0.8)

        return pos_i_com
 

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.inference(users, pos_items)
        neg_scores = self.inference(users, neg_items)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        return bpr_loss

    @staticmethod
    def max_norm(param, max_val=1, eps=1e-8):
        norm = param.norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norm, 0, max_val)
        param = param * (desired / (eps + norm))

        return param


    def forward(self, user, pos, neg):
        # thr: threshold

        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = \
            self.user_embeddings(user_id), self.item_embeddings(pos_id), self.item_embeddings(neg_id)

        pos_i_com = self.community_ui(user_id)

        self.agg_user_embeddings.weight[user_id] = user_emb.data
        self.agg_item_embeddings.weight[pos_id] = pos_emb.data
        self.agg_item_embeddings.weight[neg_id] = neg_emb.data

        return user_emb, pos_emb, neg_emb, pos_i_com



    def out_forward(self, user, pos, neg, user_embeddings, item_embeddings, cluster_ids):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = user_embeddings[user_id], item_embeddings[pos_id], item_embeddings[neg_id]

        # compute updated cluster_center_emb
        cluster_center_emb = None
        for i in range(self.cluster):
            index = torch.where(cluster_ids == i)[0]
            tmp_emb = item_embeddings[index]
            cluster_center_emb = tmp_emb if cluster_center_emb == None else torch.cat((cluster_center_emb, tmp_emb), 0)

        pos_item_cluster_id = cluster_ids[pos_id]
        neg_item_cluster_id = cluster_ids[neg_id]
        pos_center = cluster_center_emb[pos_item_cluster_id]
        neg_center = cluster_center_emb[neg_item_cluster_id]

        return user_emb, pos_emb, neg_emb, pos_center, neg_center


    def predict(self, user_id):
        user_emb = self.user_embeddings(user_id).data
        user_emb_expanded = user_emb.reshape(user_emb.shape[0], 1, user_emb.shape[1]).repeat_interleave(self.num_items, dim=1).reshape(-1, self.dim)
        item_emb_expanded = self.item_embeddings.weight.data[:-1].reshape(1, self.num_items, self.dim).repeat_interleave(user_emb.shape[0], dim=0).reshape(-1, self.dim)

        
        pred = torch.tensor([]).to(self.device)
        batch_size = 1000
        batch_num = item_emb_expanded.shape[0] // batch_size + 1
        for i in range(batch_num):
            if i < batch_num - 1:
                pred_user = self.inference(user_emb_expanded[i*batch_size:(i+1)*batch_size], item_emb_expanded[i*batch_size:(i+1)*batch_size])
                pred = torch.concat([pred, pred_user], axis=0)
            else:
                pred_user = self.inference(user_emb_expanded[i*batch_size:], item_emb_expanded[i*batch_size:])
                pred = torch.concat([pred, pred_user], axis=0)   

        pred = pred.reshape(-1, self.num_items)  
        return pred


    def get_embeddings(self, ids, emb_name):
        if emb_name == 'user':
            return self.user_embeddings[ids]
        elif emb_name == 'item':
            return self.item_embeddings[ids]
        else:
            return None


class Controller(nn.Module):
    def __init__(self, dim1, device):
        super(Controller, self).__init__()

        self.linear1 = nn.Linear(dim1, 64, bias=True).to(device)
        self.linear2 = nn.Linear(64, 1, bias=False).to(device)

    def forward(self, x):
        z1 = torch.relu(self.linear1(x))
        # res = F.sigmoid(self.linear2(z1))
        res = F.softplus(self.linear2(z1))

        return res


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(NeuMF, self).__init__()

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.dim = args.dim
        self.train_label = args.train_label
        #self.cluster = args.cluster
        self.norm = args.norm
        #self.scale1 = args.scale1
        #self.scale2 = args.scale2
        self.layers = [128, 128, 128] #[512, 256, 128]
        self.dropout = 0.1
        
        self.mf_user_embeddings = nn.Embedding(num_embeddings=self.num_users, embedding_dim=args.dim).to(self.device)
        self.mf_item_embeddings = nn.Embedding(num_embeddings=self.num_items + 1, embedding_dim=args.dim, padding_idx=self.num_items).to(self.device)
        self.mlp_user_embeddings = nn.Embedding(num_embeddings=self.num_users, embedding_dim=args.dim).to(self.device)
        self.mlp_item_embeddings = nn.Embedding(num_embeddings=self.num_items + 1, embedding_dim=args.dim, padding_idx=self.num_items).to(self.device)
        
        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.dim
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.dim, 1, bias=False)
        
    def inference(self, mf_users_embed, mf_items_embed, mlp_users_embed, mlp_items_embed):
        mf_vector = mf_users_embed * mf_items_embed # MF part
        mlp_vector = torch.cat([mlp_users_embed, mlp_items_embed], dim=-1) # MLP part
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)
        
        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)
        
        return prediction
    
    
    def community_ui(self, user_id):
        batch_label = self.train_label[user_id]
        mf_pos_i_com = torch.matmul(batch_label, self.mf_item_embeddings.weight)
        mlp_pos_i_com = torch.matmul(batch_label, self.mlp_item_embeddings.weight)
        num = batch_label.sum(1).unsqueeze(1)
        
        mf_pos_i_com = mf_pos_i_com / num
        mlp_pos_i_com = mlp_pos_i_com / num
        

        return mf_pos_i_com, mlp_pos_i_com

    def bpr_loss(self, mf_users, mf_pos_items, mf_neg_items, mlp_users, mlp_pos_items, mlp_neg_items):
        pos_scores = self.inference(mf_users, mf_pos_items, mlp_users, mlp_pos_items)
        neg_scores = self.inference(mf_users, mf_neg_items, mlp_users, mlp_neg_items)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        return bpr_loss
    
    @staticmethod
    def max_norm(param, max_val=1, eps=1e-8):
        norm = param.norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norm, 0, max_val)
        param = param * (desired / (eps + norm))

        return param
    
    def forward(self, user, pos, neg):
        # thr: threshold

        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        mf_user_emb, mf_pos_emb, mf_neg_emb = \
            self.mf_user_embeddings(user_id), self.mf_item_embeddings(pos_id), self.mf_item_embeddings(neg_id)
            
        mlp_user_emb, mlp_pos_emb, mlp_neg_emb = \
            self.mlp_user_embeddings(user_id), self.mlp_item_embeddings(pos_id), self.mlp_item_embeddings(neg_id)

        mf_pos_i_com, mlp_pos_i_com = self.community_ui(user_id)

        return mf_user_emb, mf_pos_emb, mf_neg_emb, mf_pos_i_com, mlp_user_emb, mlp_pos_emb, mlp_neg_emb, mlp_pos_i_com
    
    #def out_forward(self, user, pos, neg, mf_user_embeddings, mf_item_embeddings, mlp_user_embeddings, mlp_item_embeddings, cluster_ids):
    def out_forward(self, user, pos, neg, mf_user_embeddings, mf_item_embeddings, mlp_user_embeddings, mlp_item_embeddings):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        mf_user_emb, mf_pos_emb, mf_neg_emb = mf_user_embeddings[user_id], mf_item_embeddings[pos_id], mf_item_embeddings[neg_id]
        mlp_user_emb, mlp_pos_emb, mlp_neg_emb = mlp_user_embeddings[user_id], mlp_item_embeddings[pos_id], mlp_item_embeddings[neg_id]
        
        """
        # for multi-interest NeuMF
        # compute updated cluster_center_emb
        cluster_center_emb = None
        for i in range(self.cluster):
            index = torch.where(cluster_ids == i)[0]
            tmp_emb = item_embeddings[index]
            cluster_center_emb = tmp_emb if cluster_center_emb == None else torch.cat((cluster_center_emb, tmp_emb), 0)

        pos_item_cluster_id = cluster_ids[pos_id]
        neg_item_cluster_id = cluster_ids[neg_id]
        pos_center = cluster_center_emb[pos_item_cluster_id]
        neg_center = cluster_center_emb[neg_item_cluster_id]
        
        return user_emb, pos_emb, neg_emb, pos_center, neg_center
        """
        return mf_user_emb, mf_pos_emb, mf_neg_emb, mlp_user_emb, mlp_pos_emb, mlp_neg_emb

    def predict(self, user_id):
        mf_user_emb = self.mf_user_embeddings(user_id).data.cpu()
        mlp_user_emb = self.mlp_user_embeddings(user_id).data.cpu()
        mf_item_embeddings = self.mf_item_embeddings.weight.data.cpu()
        mlp_item_embeddings = self.mlp_item_embeddings.weight.data.cpu()
        
        mf_user_emb_expanded = mf_user_emb.reshape(mf_user_emb.shape[0], 1, mf_user_emb.shape[1]).repeat_interleave(self.num_items, dim=1).reshape(-1, self.dim)
        mlp_user_emb_expanded = mlp_user_emb.reshape(mf_user_emb.shape[0], 1, mlp_user_emb.shape[1]).repeat_interleave(self.num_items, dim=1).reshape(-1, self.dim)
        mf_item_emb_expanded = mf_item_embeddings[:-1].reshape(1, self.num_items, self.dim).repeat_interleave(mf_user_emb.shape[0], dim=0).reshape(-1, self.dim)
        mlp_item_emb_expanded = mlp_item_embeddings[:-1].reshape(1, self.num_items, self.dim).repeat_interleave(mlp_user_emb.shape[0], dim=0).reshape(-1, self.dim)
        #mf_item_emb_expanded = self.mf_item_embeddings.weight.data[:-1].reshape(1, self.num_items, self.dim).repeat_interleave(mf_user_emb.shape[0], dim=0).reshape(-1, self.dim)
        #mlp_item_emb_expanded = self.mlp_item_embeddings.weight.data[:-1].reshape(1, self.num_items, self.dim).repeat_interleave(mlp_user_emb.shape[0], dim=0).reshape(-1, self.dim)

        
        #pred = torch.tensor([]).to(self.device)
        pred = torch.tensor([])
        batch_size = 3000000 # amazon-cds::1000000
        batch_num = mf_item_emb_expanded.shape[0] // batch_size + 1
        print('batch_num', batch_num)
        for i in range(batch_num):
            print('i', i)
            if i < batch_num - 1:
                pred_user = self.inference(mf_user_emb_expanded[i*batch_size:(i+1)*batch_size].to(self.device), mf_item_emb_expanded[i*batch_size:(i+1)*batch_size].to(self.device), mlp_user_emb_expanded[i*batch_size:(i+1)*batch_size].to(self.device), mlp_item_emb_expanded[i*batch_size:(i+1)*batch_size].to(self.device))
                pred = torch.concat([pred, pred_user.detach().cpu()], axis=0)
            else:
                pred_user = self.inference(mf_user_emb_expanded[i*batch_size:].to(self.device), mf_item_emb_expanded[i*batch_size:].to(self.device), mlp_user_emb_expanded[i*batch_size:].to(self.device), mlp_item_emb_expanded[i*batch_size:].to(self.device))
                pred = torch.concat([pred, pred_user.detach().cpu()], axis=0)   

        pred = pred.reshape(-1, self.num_items)  
       
        
        return pred
        
    
    def get_embeddings(self, ids, emb_name):
        if emb_name == 'mf_user':
            return self.mf_user_embeddings[ids]
        elif emb_name == 'mf_item':
            return self.mf_item_embeddings[ids]
        elif emb_name == 'mlp_user':
            return self.mlp_user_embeddings[ids]
        elif emb_name == 'mlp_item':
            return self.mlp_item_embeddings[ids]
        else:
            return None        
        