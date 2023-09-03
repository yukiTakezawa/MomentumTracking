import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import time
import random


class DecentLaMOptimizer(Optimizer):
    def __init__(self, params, node_id: int, adj_node_ids: list, lr=1e-5, beta=0.9, device="cuda"):

        self.node_id = node_id
        self.adj_node_ids = adj_node_ids
        self.device = device
        self.counter = dict.fromkeys(self.adj_node_ids, 0)
        self.l2_penalty = 0.001
        self.beta = beta
        self.lr = lr
        self.n_sent_params = 0
        
        defaults = dict(lr=lr, beta=self.beta)
        super(DecentLaMOptimizer, self).__init__(params, defaults)
        
        # generate initial dual variables.
        for group in self.param_groups:
            group["momentum"] = []
            group["adj_modified_grad"] = [] # x_j - lr*\nabla f_j
            group["modified_grad"] = [] # x_i - lr*\nabla f_i
            
            for p in group["params"]:
                adj_modified_grad = {}
                
                for adj_node_id in adj_node_ids:
                    adj_modified_grad[adj_node_id] = torch.zeros_like(p, device=device)
                    
                group["adj_modified_grad"].append(adj_modified_grad)
                group["modified_grad"].append(torch.zeros_like(p, device=self.device))
                group["momentum"].append(torch.zeros_like(p, device=self.device))
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        for group in self.param_groups:
            lr = group['lr']

            for p, modified_grad in zip(group["params"], group["modified_grad"]):
                modified_grad.data = p - lr * (p.grad + self.l2_penalty*p)

        self.exchange(self.adj_node_ids)

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
        
            for p, momentum, modified_grad, adj_modified_grad in zip(group['params'], group["momentum"], group["modified_grad"], group["adj_modified_grad"]):
                g = - 1/lr * modified_grad
                for node_id in self.adj_node_ids:
                    g -= 1/lr * adj_modified_grad[node_id]
                g /= (1 + len(self.adj_node_ids))
                g += 1/lr * p
                
                # update model parameters W_i
                momentum.data = beta * momentum + g
                p.data = p.data - lr * momentum
                
        if closure is not None:
            loss = closure()
        
        return loss


    @torch.no_grad()
    def exchange(self, node_ids):
        task_list = []
        recieved_params = {}
        for node_id in node_ids:
            task_list += self.send_param(node_id)
            tasks, params = self.recv_param(node_id)
            task_list += tasks
            recieved_params[node_id] = params
            
        for task in task_list:
            task.wait()

        self.store_param(recieved_params)

        
    @torch.no_grad()
    def send_param(self, node_id):
        task_list = []
        
        for group in self.param_groups:
            for i, modified_grad in enumerate(group["modified_grad"]):
                task_list.append(dist.isend(tensor=modified_grad.to("cpu"), dst=node_id, tag=i))

        return task_list

    
    @torch.no_grad()
    def recv_param(self, node_id):
        task_list = []
        recieved_params = []
        
        for group in self.param_groups:        
            for i, adj_modified_grad in enumerate(group["adj_modified_grad"]):    
                tmp = torch.zeros_like(adj_modified_grad[node_id], device="cpu")
                task_list.append(dist.irecv(tensor=tmp, src=node_id, tag=i))
                #adj_p[node_id].data = tmp.data.to(self.device)
                recieved_params.append(tmp)
                
        return task_list, recieved_params

    
    @torch.no_grad()
    def store_param(self, recieved_params):
        for group in self.param_groups:
            for i, adj_modified_grad in enumerate(group["adj_modified_grad"]):
                for node_id in self.adj_node_ids:
                    adj_modified_grad[node_id].data = recieved_params[node_id][i].to(self.device)

                    
    @torch.no_grad()
    def param_diff(self):
        return -1.0
