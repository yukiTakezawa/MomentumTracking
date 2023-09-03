import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from collections import defaultdict, OrderedDict
import pickle
from tqdm import tqdm
import random
import math
import argparse    
import json
import random

from timm.scheduler import * 


from model.lenet_cifar10 import *
from model.resnet_cifar10 import *
from model.vgg_cifar10 import *

from optimizer.gossip_optimizer import *
from optimizer.qg_dsgdm_optimizer import *
from optimizer.gradient_tracking_optimizer import *
from optimizer.decentlam_optimizer import *
from optimizer.momentum_tracking_optimizer import *

from data.loader import *

torch.backends.cudnn.benchmark = True

def run(rank, size, datasets, config):
    # initialize the model parameters with same seed value.
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    
    torch.set_num_threads(1)

    if config["model"] == "lenet":
        net = LeNetCifar10(device=config["device"][rank]).to(config["device"][rank])
    elif config["model"] == "resnet":
        net = ResNetCifar10(device=config["device"][rank]).to(config["device"][rank])
    elif config["model"] == "vgg":
        net = VggCifar10(device=config["device"][rank]).to(config["device"][rank])
        
    net.to(config["device"][rank])
    
    loaders = datasets_to_loaders(datasets, config["batch"])

    if config["method"] == "gossip":
        optimizer = GossipOptimizer(params=net.parameters(), node_id=rank, adj_node_ids=config["nw"][rank], lr=config["lr"], device=config["device"][rank], beta=0.0)
    elif config["method"] == "dsgdm":
        optimizer = GossipOptimizer(params=net.parameters(), node_id=rank, adj_node_ids=config["nw"][rank], lr=config["lr"], device=config["device"][rank], beta=config["momentum"])
    elif config["method"] == "decentlam":
        optimizer = DecentLaMOptimizer(params=net.parameters(), node_id=rank, adj_node_ids=config["nw"][rank], lr=config["lr"], device=config["device"][rank], beta=config["momentum"])
    elif config["method"] == "qg_dsgdm":
        optimizer = QgDsgdmOptimizer(params=net.parameters(), node_id=rank, adj_node_ids=config["nw"][rank], lr=config["lr"], device=config["device"][rank], beta=config["momentum"])
    elif config["method"] == "gradient_tracking":
        optimizer = GTOptimizer(params=net.parameters(), node_id=rank, adj_node_ids=config["nw"][rank], lr=config["lr"], device=config["device"][rank], beta=0.0)
    elif config["method"] == "momentum_tracking":
        optimizer = MTOptimizer(params=net.parameters(), node_id=rank, adj_node_ids=config["nw"][rank], lr=config["lr"], device=config["device"][rank], beta=config["momentum"])


    if config["model"] != "lenet":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(config["epochs"]/2), int(config["epochs"] * 3 / 4)], gamma=0.1)
        
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [], "diff_param": []}
    history["all_train_loss"]  = []
    history["all_train_acc"] = []

    count_epoch = 0

    
    with tqdm(range(config["epochs"]), desc=("node "+str(rank)), position=rank) as pbar:
        for epoch in pbar:
            
            train_loss, train_acc = net.run(loaders, optimizer)

            if (count_epoch % 10 == 0) or (count_epoch == config["epochs"] -1):
                val_loss, val_acc = net.run_val(loaders)
                #all_train_loss, all_train_acc = net.run_all_train(loaders)
                test_loss, test_acc = net.run_test(loaders)
                
                # save loss and accuracy
                history["train_loss"] += [train_loss]
                history["test_loss"] += [test_loss]
                history["val_loss"] += [val_loss]
                history["train_acc"] += [train_acc]
                history["test_acc"] += [test_acc]
                history["val_acc"] += [val_acc]
            
                #history["all_train_loss"] += [all_train_loss]
                #history["all_train_acc"] += [all_train_acc]
                
            
                history["diff_param"].append(optimizer.param_diff())
            
                pbar.set_postfix(OrderedDict(loss=(round(train_loss, 2), round(test_loss, 2)), acc=(round(train_acc, 2), round(test_acc, 2)), diff=(history["diff_param"][-1])))
                
            count_epoch += 1

            if config["model"] != "lenet":
                #scheduler.step(count_epoch)
                scheduler.step()
                
    pickle.dump(history, open(config["log_path"] + "node" + str(rank) + ".pk", "wb"))
    
    
def init_process(rank, size, datasets, config, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = config["port"] #'29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, datasets, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PowerECL')
    parser.add_argument('method', default="powerecl", type=str)
    parser.add_argument('log', default="./log/powerecl", type=str)
    parser.add_argument('--batch', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default="lenet", type=str)
    parser.add_argument('--port', default='29500', type=str)
    parser.add_argument('--nw', default="config/ring3_iid.json", type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', default=None, type=str) # if None, use "nw" file parameter.
    parser.add_argument('--epoch', default=1000, type=int)
    args = parser.parse_args()

    config = defaultdict(dict)
    config["lr"] = args.lr
    config["seed"] = args.seed 
    config["momentum"] = args.momentum   
    config["epochs"] = args.epoch
    config["log_path"] = args.log
    config["method"] = args.method
    config["port"] = args.port
    config["batch"] = args.batch
    config["model"] = args.model
    
    config_json = json.load(open(args.nw, "r"))
    
    n_node = len(config_json)
    
    config["nw"] = [config_json["node" + str(i)]["adj"] for i in range(n_node)]
    config["node_label"] = [config_json["node" + str(i)]["n_class"] for i in range(n_node)]

    if args.cuda is None:
        config["device"] = [config_json["node" + str(i)]["cuda"] for i in range(n_node)]
    else:
        config["device"] = [args.cuda for _ in range(n_node)]

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    datasets = load_CIFAR10_hetero(config["node_label"], batch=config["batch"], val_rate=0.1)
    
    processes = []
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    for rank in range(n_node):
        node_datasets = {"train": datasets["train"][rank], "val": datasets["val"], "all_train": datasets["all_train"], "test": datasets["test"]}
        p = mp.Process(target=init_process, args=(rank, n_node, node_datasets, config, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
