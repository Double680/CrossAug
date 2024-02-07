import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import wandb
import argparse
import random
import sys
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import *
from models import *
from utils import *


def init():
    # basic configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', default='Movie-Music', help='Movie-Music/Cell-Elec', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--neg_valid_num', default=499, type=int)
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--n_hh', default=8, type=int)
    parser.add_argument('--l_alpha', default=0.7, type=float)
    parser.add_argument('--cd_alpha', default=0.7, type=float)
    parser.add_argument('--ll', default=1, type=int)
    parser.add_argument('--cdl', default=1, type=float)
    parser.add_argument('--al', default=0.5, type=float)

    args = parser.parse_args()

    args.dataset = "Amazon"
    assert args.domains in ["Movie-Music", "Cell-Elec"]
    args.approach = "CrossAug"

    # set device
    device = set_device(args.gpu)
    args.device = device

    # set seed
    set_seed(args.seed)

    # configuration
    with open('config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    load_data(args, config["datasets"][args.dataset][args.domains])
    load_models(args, config[args.approach])

    if args.wandb:
        name = "Ours"
        if args.ll:
            name = name + f",LL={args.ll}"
        if args.cdl:
            name = name + f",CDL={args.cdl}"
        if args.al:
            name = name + f",AL={args.al},HH={args.n_hh}"
            
        wandb.init(
            project=f"DTCDR-{args.dataset}-{args.domains}-{args.approach}",
            name=name
        )

    return args


# load models

def load_models(args, config):
    print(f"Model: {args.approach}.")

    args.params = config
    args.lr = 0.001 if args.domains == "Movie-Music" else 0.0005
    args.model = torch.compile(getattr(sys.modules[__name__], args.approach)(args).to(args.device))
    args.optim = getattr(sys.modules[__name__], config["optim"])(args.model.parameters(), lr=args.lr)

    
def train(args):   
    evaluate(args, 0, "valid") 

    print("Constructing negative dict...")

    d1_user2count, d1_user2item = construct_inter(args.d1)
    d2_user2count, d2_user2item = construct_inter(args.d2)

    d1_train = np.array(args.d1["train"])
    d1_train = d1_train[np.argsort(d1_train[:, 0])]
    d2_train = np.array(args.d2["train"])
    d2_train = d2_train[np.argsort(d2_train[:, 0])]

    total_len = len(d1_train) + len(d2_train)
    index_list = [item for item in range(total_len)]
    batch_size = args.params["batch_size"]
    n_epoch = args.params["epoch"]

    for i in range(n_epoch):
        args.model.train()

        print(f"TRAIN - Epoch {i+1}")

        d1_neg = construct_neg(args, d1_train[:, 0], d1_user2count, d1_user2item, args.d1["n_items"])
        d2_neg = construct_neg(args, d2_train[:, 0], d2_user2count, d2_user2item, args.d2["n_items"])

        d1_index = np.arange(len(d1_train))
        np.random.shuffle(d1_index)
        d2_index = np.arange(len(d2_train))
        np.random.shuffle(d2_index)

        d1_pos, d1_neg = d1_train[d1_index], d1_neg[d1_index]
        d2_pos, d2_neg = d2_train[d2_index], d2_neg[d2_index]

        d1_point, d2_point = 0, 0
        random.shuffle(index_list)
        
        for batch in range(0, total_len, batch_size):
            indecies = np.array(index_list[batch : min(batch+batch_size, total_len)])
            d1_num, d2_num = np.sum(indecies < len(d1_train)), np.sum(indecies >= len(d1_train))
            
            d1_inter = torch.from_numpy(d1_pos[d1_point:d1_point+d1_num]).to(args.device)
            d1_neg_item = torch.from_numpy(d1_neg[d1_point:d1_point+d1_num]).to(args.device)
            d2_inter = torch.from_numpy(d2_pos[d2_point:d2_point+d2_num]).to(args.device)
            d2_neg_item = torch.from_numpy(d2_neg[d2_point:d2_point+d2_num]).to(args.device)

            args.optim.zero_grad()
            loss = args.model.calculate_loss(d1_inter, d1_neg_item, d2_inter, d2_neg_item)            
            loss.backward()
            args.optim.step()

            if batch % 100 == 0:
                print(f"{batch/total_len*100:8.2f}%  Loss: {loss.item():.4f}")
                if args.wandb:
                    wandb.log({"loss": loss.item()})

            d1_point, d2_point = d1_point + d1_num, d2_point + d2_num    

        if (i+1) % 5 == 0:
            evaluate(args, i+1, "valid")

    print("Training process finished!")
 

def evaluate(args, epoch=None, mode="test", eval_size=32):
    args.model.eval()
    
    if mode == "valid":
        print(f"VALID - Epoch {epoch}")
    else:
        print(f"TEST")

    eval_set_1 = DataLoader(TensorDataset(torch.tensor(args.d1[mode]).to(args.device)), batch_size=eval_size)
    eval_set_2 = DataLoader(TensorDataset(torch.tensor(args.d2[mode]).to(args.device)), batch_size=eval_size)

    args.model.predict(eval_set_1, eval_set_2, mode)


if __name__ == '__main__':
    args = init()
    train(args)
    evaluate(args)
