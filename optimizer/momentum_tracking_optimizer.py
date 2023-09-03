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


class MTOptimizer(Optimizer):
    def __init__(self, params, node_id: int, adj_node_ids: list, lr=1e-5, beta=0.9, n_nodes=8, device="cuda"):
        self.lr = lr
        self.node_id = node_id
        self.adj_node_ids = adj_node_ids
        self.device = device
        self.beta = beta
        self.l2_penalty = 0.001
        
        self.initialization_flag = True
        self.n_nodes = n_nodes
        
        defaults = dict(lr=lr, beta=beta)

        super(MTOptimizer, self).__init__(params, defaults)

        # generate initial dual variables.
        for group in self.param_groups:
            group["dual_c"] = []
            group["adj_c"] = []
            group["adj_params"] = []
            group["momentum"] = []
            
            for p in group["params"]:
                adj_c = {}
                adj_params = {}

                for adj_node_id in adj_node_ids:
                    adj_params[adj_node_id] = p.clone().detach() #torch.zeros_like(p, device=self.device)
                    adj_c[adj_node_id] = torch.zeros_like(p, device=self.device) # dual_c - grad

                group["dual_c"].append(torch.zeros_like(p, device=self.device))
                group["momentum"].append(torch.zeros_like(p, device=self.device))
                group["adj_c"].append(adj_c)
                group["adj_params"].append(adj_params)

    @torch.no_grad()
    def initialize(self):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if self.initialization_flag:
            for group in self.param_groups:
                beta = group['beta']
                
                for p, dual_c, momentum in zip(group['params'], group['dual_c'], group["momentum"]):
                    initial_grad = p.grad.clone()
                    dist.all_reduce(initial_grad, op=dist.ReduceOp.SUM)
                    initial_grad /= self.n_nodes
                    dual_c.data = (p.grad - initial_grad) / (1 - beta)
                    momentum.data = (p.grad - initial_grad) / (1 - beta)

            self.initialization_flag = False
        
        # update momentum
        for group in self.param_groups:
            beta = group['beta']

            for p, momentum in zip(group['params'], group['momentum']):
                momentum.data = beta * momentum + (p.grad + self.l2_penalty * p)


        # Computes average with its neighbors.
        self.exchange()
        
        for group in self.param_groups:
            lr = group['lr']
                    
            for p, momentum, dual_c in zip(group['params'], group['momentum'], group["dual_c"]):
                #p.data = p.data - lr * (p.grad.data - dual_c.data)
                p.data = p.data - lr * (momentum - dual_c)

        self.update_dual()
        
        if closure is not None:
            loss = closure()

        return loss


    @torch.no_grad()
    def exchange(self):
        task_list = []
        recieved_params = {}
        
        for node_id in self.adj_node_ids:
            task_list += self.send_param(node_id)
            tasks, params = self.recv_param(node_id)
            task_list += tasks
            recieved_params[node_id] = params

        for task in task_list:
            task.wait()

        self.average_param(recieved_params)
        task_list = []
        recieved_duals = {}
        
        for node_id in self.adj_node_ids:
            task_list += self.send_dual(node_id)
            tasks, duals = self.recv_dual(node_id)
            task_list += tasks
            recieved_duals[node_id] = duals
            
        for task in task_list:
            task.wait()

        self.store_dual(recieved_duals)
        
        """
            if self.node_id < node_id:
                self.send_dual(node_id)
                self.recv_dual(node_id)
                self.send_param(node_id)
                self.recv_param(node_id)
            else:
                self.recv_dual(node_id)
                self.send_dual(node_id)
                self.recv_param(node_id)
                self.send_param(node_id)
            """

        
            
    @torch.no_grad()
    def send_param(self, node_id):
        task_list = []
        
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                task_list.append(dist.isend(tensor=p.data.to("cpu"), dst=node_id, tag=i))
        return task_list
    

    @torch.no_grad()
    def recv_param(self, node_id):
        task_list = []
        recieved_params = []
        
        for group in self.param_groups:
            for i, adj_p in enumerate(group["adj_params"]):
                tmp = torch.zeros_like(adj_p[node_id], device="cpu")
                task_list.append(dist.irecv(tensor=tmp, src=node_id, tag=i))
                recieved_params.append(tmp)

        return task_list, recieved_params
            

    @torch.no_grad()
    def average_param(self, recieved_params):
        for group in self.param_groups:
            for i, (p, adj_p) in enumerate(zip(group["params"], group["adj_params"])):
                for node_id in self.adj_node_ids:
                    adj_p[node_id].data = recieved_params[node_id][i].to(self.device)
                    p.data += adj_p[node_id]
                p.data /= (len(self.adj_node_ids) + 1)

                
    @torch.no_grad()
    def send_dual(self, node_id):
        """
        Send dual_c - grad.
        """
        task_list = []
        
        for group in self.param_groups:
            for i, (momentum, dual_c) in enumerate(zip(group["momentum"], group["dual_c"])):
                task_list.append(dist.isend(tensor=(dual_c - momentum).to("cpu"), dst=node_id, tag=i))
        return task_list
    

    @torch.no_grad()
    def recv_dual(self, node_id):
        """
        Receive dual_c - grad, and store it.
        """
        task_list = []
        recieved_dual = []
        
        for group in self.param_groups:
            for i, adj_c in enumerate(group["adj_c"]):
                tmp = torch.zeros_like(adj_c[node_id], device="cpu")
                task_list.append(dist.irecv(tensor=tmp, src=node_id, tag=i))
                #adj_c[node_id].data = tmp.to(self.device)
                recieved_dual.append(tmp)
        return task_list, recieved_dual


    @torch.no_grad()
    def store_dual(self, recieved_duals):
        for group in self.param_groups:
            for i, adj_c in enumerate(group["adj_c"]):
                for node_id in self.adj_node_ids:
                    adj_c[node_id].data = recieved_duals[node_id][i].to(self.device)


    @torch.no_grad()
    def update_dual(self):
        for group in self.param_groups:

            for i, (momentum, dual_c, adj_c) in enumerate(zip(group["momentum"], group["dual_c"], group["adj_c"])):

                tmp = torch.zeros_like(dual_c)
                for node_id in self.adj_node_ids:
                    tmp += adj_c[node_id] # adj_c[node_id] is dual_c - grad.
                tmp += dual_c - momentum

                dual_c.data = tmp/(1 + len(self.adj_node_ids)) + momentum


    @torch.no_grad()
    def param_diff(self):
        diff = 0.
        for group in self.param_groups:
            for i, (p, adj_p) in enumerate(zip(group["params"], group["adj_params"])):
                for node_id in self.adj_node_ids:
                    diff += torch.norm(p - adj_p[node_id]).detach().cpu()
        return diff
