import os
import torch
import scipy.sparse as sp
import numpy as np
import random
from collections import defaultdict


# load data

def load_tri(path, mode):
    with open(os.path.join(path, f"{mode}.txt"), "r") as file:
        data = file.readlines()[1:]
        data = [tri.strip('\n').split('\t') for tri in data]
        data = [list(map(int, tri)) for tri in data]

    if mode == "train":
        train_data = []
        for item in data:
            if item[2] == 1:
                train_data.append(item)
        data = train_data

    return data


def get_inter_mat(args):
    d1_n_users, d2_n_users = args.d1['n_users'], args.d2['n_users']
    n_users = d1_n_users + d2_n_users - args.n_shared_users
    d1_n_items, d2_n_items = args.d1['n_items'], args.d2['n_items']
    n_items = d1_n_items + d2_n_items

    check_paths(f"materials/{args.dataset}", f"materials/{args.dataset}/{args.domains}")
    path_root = f"materials/{args.dataset}/{args.domains}"
    
    if os.path.exists(f"{path_root}/d1_inters.npy") and os.path.exists(f"{path_root}/d1_inters_ui.npy") and \
        os.path.exists(f"{path_root}/d2_inters.npy") and os.path.exists(f"{path_root}/d2_inters_ui.npy") and \
        os.path.exists(f"{path_root}/overall_inters.npy"):

        d1_inters = np.load(f"{path_root}/d1_inters.npy")
        row, col, data = d1_inters[:, 0], d1_inters[:, 1], np.ones(len(d1_inters))
        args.d1['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d1_n_items, n_users+d1_n_items))

        d1_inters_ui = np.load(f"{path_root}/d1_inters_ui.npy")
        row, col, data = d1_inters_ui[:, 0], d1_inters_ui[:, 1], np.ones(len(d1_inters_ui))
        args.d1['inter_mat_ui'] = sp.coo_matrix((data, (row, col)), shape=(n_users, d1_n_items))

        d2_inters = np.load(f"{path_root}/d2_inters.npy")
        row, col, data = d2_inters[:, 0], d2_inters[:, 1], np.ones(len(d2_inters))
        args.d2['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d2_n_items, n_users+d2_n_items))

        d2_inters_ui = np.load(f"{path_root}/d2_inters_ui.npy")
        row, col, data = d2_inters_ui[:, 0], d2_inters_ui[:, 1], np.ones(len(d2_inters_ui))
        args.d2['inter_mat_ui'] = sp.coo_matrix((data, (row, col)), shape=(n_users, d2_n_items))

        overall_inters = np.load(f"{path_root}/overall_inters.npy")
        row, col, data = overall_inters[:, 0], overall_inters[:, 1], np.ones(len(overall_inters))
        args.overall_mat = sp.coo_matrix((data, (row, col)), shape=(n_users+n_items, n_users+n_items))

        return

    # construct adj matrix for domain 1
    d1_inters, d1_inters_ui = [], []
    for inter in args.d1['train']:
        if inter[2] == 1:
            user, item = inter[0], inter[1]
            d1_inters.append((user, item))
            d1_inters_ui.append((user, item))
    d1_inters = [(u, v + n_users) for (u, v) in d1_inters]
    d1_overall = d1_inters
    d1_inters += [(v, u) for (u, v) in d1_inters]
    d1_overall += [(v, u) for (u, v) in d1_overall]

    d1_n_inters = len(d1_inters)
    d1_n_inters_ui = len(d1_inters_ui)

    d1_inters = np.array(d1_inters)
    np.save(f"{path_root}/d1_inters.npy", d1_inters)
    row, col, data = d1_inters[:, 0], d1_inters[:, 1], np.ones(d1_n_inters)
    args.d1['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d1_n_items, n_users+d1_n_items))

    d1_inters_ui = np.array(d1_inters_ui)
    np.save(f"{path_root}/d1_inters_ui.npy", d1_inters_ui)
    row, col, data = d1_inters_ui[:, 0], d1_inters_ui[:, 1], np.ones(d1_n_inters_ui)
    args.d1['inter_mat_ui'] = sp.coo_matrix((data, (row, col)), shape=(n_users, d1_n_items))

    # construct adj matrix for domain 2
    d2_inters, d2_inters_ui = [], []
    for inter in args.d2['train']:
        if inter[2] == 1:
            user, item = inter[0], inter[1]
            if user < args.n_shared_users:
                d2_inters.append((user, item))
                d2_inters_ui.append((user, item))
            else:
                d2_inters.append((user+args.d1['n_users']-args.n_shared_users, item))
                d2_inters_ui.append((user+args.d1['n_users']-args.n_shared_users, item))
            
    d2_inters = [(u, v + n_users) for (u, v) in d2_inters]
    d2_overall = [(u, v + d1_n_items) for (u, v) in d2_inters]
    d2_inters += [(v, u) for (u, v) in d2_inters]
    d2_overall += [(v, u) for (u, v) in d2_overall]

    d2_n_inters = len(d2_inters)
    d2_n_inters_ui = len(d2_inters_ui)

    d2_inters = np.array(d2_inters)
    np.save(f"{path_root}/d2_inters.npy", d2_inters)
    row, col, data = d2_inters[:, 0], d2_inters[:, 1], np.ones(d2_n_inters)
    args.d2['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_users+d2_n_items, n_users+d2_n_items))

    d2_inters_ui = np.array(d2_inters_ui)
    np.save(f"{path_root}/d2_inters_ui.npy", d2_inters_ui)
    row, col, data = d2_inters_ui[:, 0], d2_inters_ui[:, 1], np.ones(d2_n_inters_ui)
    args.d2['inter_mat_ui'] = sp.coo_matrix((data, (row, col)), shape=(n_users, d2_n_items))


# construct training set
def construct_train(config, user2count, user2inter, neg_ratio):
    data = []
    data += config["train"]
    indices = [i for i in range(config["n_items"])]
    for user in range(config["n_users"]):
        neg = random.sample(indices, user2count[user] * neg_ratio + len(user2inter[user]))
        for item in neg:
            if item not in user2inter[user]:
                data.append([user, item, 0])

    random.shuffle(data)
    data = np.array(data)
    return data


def construct_neg(args, train_user, user2count, user2inter, n_items):
    neg_ratio = args.params["neg_ratio"]
    total_neg_num = len(train_user) * neg_ratio
    neg_pool = []
    cnt, user, next_user_cnt = 0, -1, 0
    while cnt < total_neg_num:
        neg_data = np.random.randint(0, n_items, size=(total_neg_num - cnt) * 2)
        for i in range(len(neg_data)):
            if cnt == next_user_cnt:
                user += 1
                next_user_cnt += user2count[user] * neg_ratio
            if neg_data[i] not in user2inter[user]:
                neg_pool.append(neg_data[i])
                cnt += 1
            if cnt == total_neg_num:
                break

    neg_pool = np.array(neg_pool)
    neg_pool = np.reshape(neg_pool, (len(train_user), neg_ratio))
    return neg_pool


def construct_inter(config):
    tr = config["train"]
    user2count = defaultdict(int)
    user2item = defaultdict(list)

    for i in range(len(tr)):
        user, item = tr[i][0], tr[i][1]
        user2count[user] += 1
        user2item[user].append(item)

    return user2count, user2item


def save_material(args, obj, path):
    torch.save(obj, os.path.join(args.material_path, path))
    

def load_material(args, path):
    obj = torch.load(os.path.join(args.material_path, path), map_location=args.device)
    return obj


def load_data(args, config):
    print(f"Dataset: {args.dataset}.  Domains: {args.domains}.")

    args.n_shared_users = config["n_shared_users"]
    args.d1 = config["domain_1"]
    args.d2 = config["domain_2"]
    
    # load train/valid/test
    path_1 = os.path.join(config['path'], config['domain_1']['name'])
    args.d1["train"] = load_tri(path_1, "train")
    args.d1["valid"] = load_tri(path_1, "valid")
    args.d1["test"] = load_tri(path_1, "test")

    path_2 = os.path.join(config['path'], config['domain_2']['name'])
    args.d2["train"] = load_tri(path_2, "train")
    args.d2["valid"] = load_tri(path_2, "valid")
    args.d2["test"] = load_tri(path_2, "test") 

    args.material_path = os.path.join("./materials", args.dataset, args.domains)

    get_inter_mat(args)

    args.data = config   
    args.torch_type = torch.float64


# Set device

def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    

# Set seed
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# Check path

def check_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)