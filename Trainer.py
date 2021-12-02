import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import eval_Z
from torch_geometric.utils import structured_negative_sampling 

class Trainer(object):
    def __init__(self, model, X, edge_index, args):
        self.model = model
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.lr,
                                          weight_decay=1e-8)

        self.bestResult = -1
        self.beta = args.beta
        self.gamma = args.gamma
        self.sigma = args.sigma

        self.X = X
        self.edge_index = edge_index

        self.node_num, self.attr_num = X.shape

        self.sim_loss_m = args.m

    def train_mini_batch(self):
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def tf_loss(self):
        attr_emb = self.model.encoder(torch.eye(self.attr_num).to(self.X.device))
        return (attr_emb.norm(dim=1) * self.X.sum(dim=0)).mean()

    def ae_loss(self):
        # return  (self.X - self.model.decoder(self.model.encoder(self.X))).norm()/self.X.shape[0]
        return F.mse_loss(self.X, self.model.decoder(self.model.encoder(self.X)))

    def sim_loss(self):
        node_emb = self.model.forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()

    def sage_sim_loss(self):
        node_emb = self.model.sage_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()

    def train_batch(self):
        self.model.train()
        loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_sage(self):
        self.model.train()
        loss = self.gamma*self.sage_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, qX, ans, topk, verbose=False):
        self.model.eval()

        node_emb = self.model.forward(self.X, self.edge_index)
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit

    def sage_test(self, qX, ans, topk, verbose=False):
        self.model.eval()

        node_emb = self.model.sage_forward(self.X, self.edge_index)
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit

    def save(self, dir):
        if dir is not None:
            torch.save(self.model.state_dict(), dir)

    def decay_learning_rate(self, epoch, init_lr):
        lr = init_lr / (1 + 0.05 * epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer


