import os
import numpy as np 
import pickle as pkl
from tqdm.notebook import tqdm

import torch
from torch.utils.data import RandomSampler, DataLoader

from .utils import accuracy, error   
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k

device = ('cuda' if torch.cuda.is_available() else 'cpu')
class Triplet_Shap(object):
    
    def __init__(self, model, sampler, val_user_list,
                 directory=None, seed=10):
        """
        Args:
            model: Torch model
            sampler: BPR Triplet Sampler
            val_user_list: Validation User List
            directory: Directory to save results and figures.
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
        """
            
        if seed is not None:
            np.random.seed(seed)

        self.directory = directory
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)  

        self.model = model
        self.sampler = sampler
        self.user_pos_item_pairs = np.asarray(self.sampler.train_matrix.todok().nonzero()).T
        self.neg_items_per_pair = np.asarray(self.sampler.pre_samples['user_neg_items'])
        self.triplets = np.concatenate([np.repeat(self.user_pos_item_pairs[:, np.newaxis, :], self.neg_items_per_pair.shape[1], axis=1).reshape(self.user_pos_item_pairs.shape[0]*self.neg_items_per_pair.shape[1],-1), \
            self.neg_items_per_pair[self.user_pos_item_pairs[:, 0]].reshape(-1,1)], axis=1)
        self.val_user_list = np.concatenate([np.arange(0, len(val_user_list)).reshape(-1, 1), val_user_list], axis=1)
        self.train_len = self.user_pos_item_pairs.shape[0]*self.neg_items_per_pair.shape[1]

        self.mem_tmc = np.zeros((0, self.train_len))
        self.idxs_tmc = np.zeros((0, self.train_len), int)
        actual_items_matrix = torch.tensor([actual_items for _, actual_items in self.val_user_list])
        self.random_score = torch.max(torch.bincount(actual_items_matrix) / len(self.val_user_list) ).item()

        self.tmc_number = self._which_parallel(self.directory)
        self._create_results_placeholder(self.directory, self.tmc_number)

    def _create_results_placeholder(self, directory, tmc_number):
        tmc_dir = os.path.join(
            directory, 
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))

    def run(self, save_every, err, tolerance=0.01):
        """Calculates data sources(points) values.
        
        Args:
            save_every: save marginal contributions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
        """

        #self.results = {}
        tmc_run = True 
        while tmc_run:
            if error(self.mem_tmc) < err:
                tmc_run = False
            else:
                self.tmc_shap(
                    save_every, 
                    tolerance=tolerance, 
                )
                self.vals_tmc = np.mean(self.mem_tmc, 0)
            self.save_results()
        return self.vals_tmc
        
        
    def save_results(self):
        """Saves results computed so far."""
        if self.directory is None:
            return
        tmc_dir = os.path.join(
            self.directory, 
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )

        raw_list = []
        for j in range(self.mem_tmc.shape[0]):
            row_dict = { self.idxs_tmc[j][i] : self.mem_tmc[j][i] for i in range(self.mem_tmc.shape[1]) }
            raw_list.append(row_dict)

        pkl.dump(raw_list, open(tmc_dir, 'wb'))
        #pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
        #         open(tmc_dir, 'wb'))
    
    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]      
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0' 
        return tmc_number

    def tmc_shap(self, iterations, tolerance=0.01):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
        """
        self._tol_mean_score()
        
        marginals, idxs = [], []
        for _ in tqdm(range(iterations)):

            marginals, idxs = self.one_iteration(
                tolerance=tolerance
            )
            self.mem_tmc = np.concatenate([
                self.mem_tmc, 
                np.reshape(marginals, (1,-1))  # dims: (some number, # train samples)
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc, 
                np.reshape(idxs, (1,-1))
            ])


    def one_iteration(self, tolerance):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs = np.random.permutation(self.train_len)                              #Re read algorithm. We can get random sampler with a dataloader instead
        marginal_contribs = np.zeros(self.train_len)

        truncation_counter = 0
        new_score = self.random_score
        self.model.train()
        pred_list = self.generate_pred_list(self.model, self.sampler.train_matrix, topk=20)

        #  Iterates through the entire Training dataset
        user_list = []
        pos_item_list = []
        for i, idx in enumerate(idxs):
            old_score = new_score
            user_list.append(self.triplets[idx][0])
            pos_items_list.append(torch.tensor(self.sampler.train_matrix[self.triplets[idx][0]]))
            if i == 0:
                user = self.triplets[idx][0].unsqueeze(0)
                pos_items_list = torch.tensor([self.sampler.train_matrix[self.triplets[idx][0]]])
            else:
                user = torch.stack(user_list, 0)
                pos_items = torch.stack(pos_items_list, 0)

            user, pos_items = user.to(device), pos_items.to(device)
            precision, recall, MAP, ndcg = self.compute_metrics(pos_items, pred_list[user], topk=20)
            new_score = recall

            marginal_contribs[idx] = (new_score - old_score)  # original code divides by 1 for some reason
            distance_to_full_score = np.abs(new_score - self.mean_score)
            #  Performance Tolerance
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs


    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.model.eval()
        pred_list = self.generate_pred_list(self.model, self.sampler.train_matrix, topk=20)
        for _ in range(100):
            #bag_idxs = np.random.choice(len(self.actual_item_matrix), len(self.actual_item_matrix))  # check size
            
            sampler = RandomSampler(self.val_user_list)
            loader = DataLoader(self.val_user_list, batch_size=512, num_workers=2, sampler=sampler)

            # 1-pass
            for user, actual_items in loader:
                user, actual_items = user.to(device), actual_items.to(device)
                precision, recall, MAP, ndcg = self.compute_metrics(actual_items, pred_list[user], topk=20)
                scores.append(recall)
                break

        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def compute_metrics(self, test_set, pred_list, topk=20):
        """Computes recommendation performance on validation set."""
        precision, recall, MAP, ndcg = [], [], [], []
        for k in [5, 10, 15, 20]:
            precision.append(precision_at_k(test_set, pred_list, k))
            recall.append(recall_at_k(test_set, pred_list, k))
            MAP.append(mapk(test_set, pred_list, k))
            ndcg.append(ndcg_k(test_set, pred_list, k))

        return precision, recall, MAP, ndcg
        
    def generate_pred_list(self, model, train_matrix, topk=20):
        num_users = train_matrix.shape[0]
        batch_size = 1024
        num_batches = int(num_users / batch_size) + 1
        user_indexes = np.arange(num_users)
        pred_list = None

        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < num_users:
                    end = num_users
                else:
                    break

            batch_user_index = user_indexes[start:end]
            batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(args.device)

            rating_pred = model.predict(batch_user_ids)
            rating_pred = rating_pred.cpu().data.numpy().copy()
            rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

            # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
            ind = np.argpartition(rating_pred, -topk)
            ind = ind[:, -topk:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

            if batchID == 0:
                pred_list = batch_pred_list
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)

        return pred_list