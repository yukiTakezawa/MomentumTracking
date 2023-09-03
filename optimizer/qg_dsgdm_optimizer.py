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


class QgDsgdmOptimizer(Optimizer):
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
        super(QgDsgdmOptimizer, self).__init__(params, defaults)
        
        # generate initial dual variables.
        for group in self.param_groups:
            group["momentum"] = []
            group["adj_params"] = []
            group["buf_params"] = [] # save the value of p to send.
            group["old_params"] = []
            
            for p in group["params"]:
                adj_params = {}
                
                for adj_node_id in adj_node_ids:
                    adj_params[adj_node_id] = torch.zeros_like(p, device=device)
                    
                group["adj_params"].append(adj_params)
                group["buf_params"].append(torch.zeros_like(p, device=self.device))
                group["old_params"].append(p.clone().detach())
                group["momentum"].append(torch.zeros_like(p, device=self.device))
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            
            for p, momentum, in zip(group['params'], group["momentum"]):

                # update model parameters W_i
                p.data = p.data - lr * (beta * momentum + (p.grad + self.l2_penalty*p))
                
        if closure is not None:
            loss = closure()

        self.update(self.adj_node_ids)
        
        return loss


    @torch.no_grad()
    def update(self, node_ids):
        """
        for node_id in node_ids:
            if self.node_id < node_id:
                self.send_param(node_id)
                self.recv_param(node_id)
                    
            else:
                self.recv_param(node_id)
                self.send_param(node_id)
        """
        task_list = []
        recieved_params = {}

        for node_id in node_ids:
            task_list += self.send_param(node_id)
            tasks, params = self.recv_param(node_id)
            task_list += tasks
            recieved_params[node_id] = params

        for task in task_list:
            task.wait()

            
        self.average_param(recieved_params)

        
    @torch.no_grad()
    def send_param(self, node_id):
        task_list = []
        
        for group in self.param_groups:
            for i, (buf_p, p) in enumerate(zip(group["buf_params"], group["params"])):
                task_list.append(dist.isend(tensor=p.to("cpu"), dst=node_id, tag=i))
                self.n_sent_params += torch.numel(p)
        return task_list
    
                
    @torch.no_grad()
    def recv_param(self, node_id):
        task_list = []
        recieved_params = []
        
        for group in self.param_groups:        
            for i, adj_p in enumerate(group["adj_params"]):    
                tmp = torch.zeros_like(adj_p[node_id], device="cpu")
                task_list.append(dist.irecv(tensor=tmp, src=node_id, tag=i))
                #adj_p[node_id].data = tmp.data.to(self.device)
                recieved_params.append(tmp)

        return task_list, recieved_params

    
    @torch.no_grad()
    def average_param(self, recieved_params):                    
                
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            
            for i, (p, adj_p, old_p, momentum) in enumerate(zip(group["params"], group["adj_params"], group["old_params"], group["momentum"])):
                
                for node_id in self.adj_node_ids:
                    adj_p[node_id].data = recieved_params[node_id][i].to(self.device)
                    p.data += adj_p[node_id]
                p.data /= (len(self.adj_node_ids) + 1)

                # update momentum
                d = (old_p -p) / lr
                momentum.data = beta * momentum + (1. - beta) * d

                # store current parameters.
                old_p.data = p
                
    @torch.no_grad()
    def param_diff(self):
        diff = 0.
        for group in self.param_groups:
            for i, (p, adj_p) in enumerate(zip(group["params"], group["adj_params"])):
                for node_id in self.adj_node_ids:
                    diff += torch.norm(p - adj_p[node_id]).detach().cpu()
        return diff
