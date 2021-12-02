import numpy as np
import scipy.sparse as sp

from scipy.sparse import lil_matrix
import torch

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_undirected

import os 
import pandas as pd


def to_str(s):
	return "'"+str(s)+"'"


def get_saved_model_path(args):
	return args.model_dir+f'/{args.dataset}{args.comment}.model'


def load_data(args):

	dataset_dir = f"./data/{args.dataset}/"

	if args.hw and args.hw!='0':
		spX_path = dataset_dir+str(args.hw)+"%hidden_spX.npz"
		print('Loading hidden attributed graph')
	else:
		spX_path = dataset_dir+"spX.npz"

	if args.he:
		edges_path = dataset_dir+f'edge_{args.he}_edge_index.npz'
	else:
		edges_path = dataset_dir+'edge_index.npz'
	
	edge_index = load_edge_index(edges_path)
	coo_X = load_spX(spX_path)

	return coo_X, edge_index


def load_spX(path):
	return sp.load_npz(path)

def load_edge_index(path):
	return np.load(path)['arr_0']

def str2int(s):
	return list(map(int,s.strip().split()))

def queries2tensor(qs, attr_num):
	q_num = len(qs)
	t = torch.zeros(q_num, attr_num)
	for i in range(q_num):
		t[i,qs[i]] = 1
	return t


def eval_Z(Z,q_emb,ans, k=100, verbose=False):
	scores = q_emb @ Z.t()
	rank = scores.sort(dim=-1, descending=True)[1]
	hits = []
	nodes_num = Z.shape[0]
	for i in range(len(q_emb)):
		mark = torch.zeros(nodes_num)
		mark[ans[i]]=1
		tmp_hit = mark[rank[i,:k]].sum()/k
		# print(f'Q_{i} hit@{k}:{tmp_hit:.4f}')
		hits.append(tmp_hit)
	hits = torch.stack(hits)
	if verbose:
		print(hits)
		res = hits.sort(descending=True)
		print('Top 30:', res[1][:30])
	return hits.mean().item()

def eval_PCA(X, qX, ans, k=100, verbose=False):
	u,s,v = torch.svd(X)
	SVD_res = eval_Z(X@v[:,:64],qX@v[:,:64],ans,k, verbose=verbose)
	return SVD_res


def coo2torch(coo):
	values = coo.data
	indices = np.vstack((coo.row, coo.col))

	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = coo.shape

	return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def load_gt(path): # gt is a dictionary {query_str:[nodes_int]}
	groud_truth = {}
	with open(path,'r') as f:
		for line in f.readlines():
			query,ans = line.strip().split('\t')
			groud_truth[query] = list(map(int,ans.split()))
	return groud_truth 

