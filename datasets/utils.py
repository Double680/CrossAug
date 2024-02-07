import os
import gzip
import json
import random
import collections
from tqdm import tqdm


# Check path

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_ratings(args):
    ratings_file_path = os.path.join(args.input_path, f"{args.domain_fullname}.csv")

    users, items, inters = set(), set(), set()
    with open(ratings_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                rating = 1 if float(rating) >= 3.5 else 0
                inters.add((user, item, rating, int(time)))
            except ValueError:
                print(line)
    
    # statistics
    print("Statistics (Original):")
    print(f"  # Users: {len(users)}\n  # Items: {len(items)}\n  # Inters: {len(inters)}\n")
    
    return inters


def load_metadata(args):
    meta_file_path = os.path.join(args.input_path, f'meta_{args.domain_fullname}.json.gz')

    items = set()
    with gzip.open(meta_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Load metadata'):
            data = json.loads(line)
            items.add(data['asin'])
    return items


def filter_inters(args, rating_inters, meta_items):
    new_inters = []

    # Get domain inters after the following timestamp:
    overtime_dict = {
        "Movie": 1.46,
        "Music": 1.42,
        "Cell": 1.43,
        "Elec": 1.48
    }
    overtime = overtime_dict[args.domain]

    # filter by meta items
    for unit in rating_inters:
        if unit[1] in meta_items and unit[3] >= int(overtime * 1e9):
            new_inters.append(unit)
    inters, new_inters = new_inters, []
    print('    The number of inters: ', len(rating_inters))

    # filter by k-core
    if args.user_k or args.item_k:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, args.user_k)
            items, n_filtered_items = generate_candidates(
                item2count, args.item_k)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    if unit[2] == 1:
                        new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
        
        args.users = sorted(users)
        args.items = sorted(items)
        args.rating_inters = inters


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        rating = unit[2]
        if rating == 1:
            user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_item_info(args):
    item_info_list = []
    already_items = set()

    features = ["asin", "title", "feature", "description", "brand", "also_buy", "also_view"]

    meta_file_path = os.path.join(args.input_path, f'meta_{args.domain_fullname}.json.gz')
    with gzip.open(meta_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Filter Item Information'):
            data = json.loads(line)
            item = data['asin']
            if item in args.items and item not in already_items:
                already_items.add(item)
                item_info = {}
                for meta_key in features:
                    if meta_key in data:
                        item_info[meta_key] = data[meta_key]
                item_info_list.append(item_info)

    args.item_info = item_info_list


def index2unit(*set_args):
    index_dict = dict()
    for unit_set in set_args:
        for unit in unit_set:
            index_dict[len(index_dict)] = unit
    return index_dict


def unit2index(index_dict):
    unit_dict = collections.defaultdict(int)
    for id in range(len(index_dict)):
        unit_dict[index_dict[id]] = id
    return unit_dict


def load_filtered_units(path, to_set=False):
    with open(path, 'r') as fp:
        units = fp.readlines()
    units = sorted(units[1:])
    units = [unit.strip('\n') for unit in units]
    if to_set:
        units = set(units)
    return units


def write_out_users(index_dict, path, overlap):
    with open(os.path.join(path, "user2index.txt"), 'w') as fp:
        fp.write(f"{len(index_dict)} {overlap}\n")
        for id in range(len(index_dict)):
            fp.write(f"{id}\t{index_dict[id]}\n")


def process_users(args):
    # load filtered users in dual domains
    print("Loading filtered users from dual domains...")

    users_path_1 = os.path.join(args.input_path, args.domain_1, "users.txt")
    users_1 = load_filtered_units(users_path_1, to_set=True)

    users_path_2 = os.path.join(args.input_path, args.domain_2, "users.txt")
    users_2 = load_filtered_units(users_path_2, to_set=True)

    # overall statistics 
    print("Summarizing statistics of integrated users...")

    joint_users = users_1 & users_2
    users_1_only = sorted(users_1 - joint_users)
    users_2_only = sorted(users_2 - joint_users)
    joint_users = sorted(joint_users)
    
    index2user_1 = index2unit(joint_users, users_1_only)
    index2user_2 = index2unit(joint_users, users_2_only)

    overlap = len(joint_users)
    args.user2index_1 = unit2index(index2user_1)
    args.user2index_2 = unit2index(index2user_2)

    print("  # Overlapping Users: ", overlap)

    # output integrated index2user files
    print("Writing out integrated users list... ", end="")

    write_out_users(index2user_1, args.output_path_1, overlap)
    write_out_users(index2user_2, args.output_path_2, overlap)

    print("Done.")


def write_out_items(index_dict, path):
    with open(os.path.join(path, "item2index.txt"), 'w') as fp:
        fp.write(f"{len(index_dict)}\n")
        for id in range(len(index_dict)):
            fp.write(f"{id}\t{index_dict[id]}\n")


def write_out_meta(meta_item, path):
    with gzip.open(os.path.join(path, "meta_item.json.gz"), 'w') as fp:
        fp.write(json.dumps(meta_item).encode('utf-8')) 


def load_filtered_meta(item2index, path):
    item_info_list = []
    with gzip.open(path, 'r') as fp:
        cnt = 0
        for line in tqdm(fp, desc='Load metadata'):
            meta_data = json.loads(line)
            for data in meta_data:
                item_info = {
                    "ID": item2index[data['asin']]
                }
                for meta_key in data:
                    if meta_key in ['also_buy', 'also_view']:
                        item_info[meta_key] = []
                        for item in data[meta_key]:
                            if item in item2index.keys():
                                item_info[meta_key].append(item2index[item])
                    else:
                        item_info[meta_key] = data[meta_key]
                item_info_list.append(item_info)
    print("Meta length:", len(item_info_list))
    return item_info_list


def process_items(args):
    # load filtered items in dual domains
    print("Loading filtered items from dual domains...")

    items_path_1 = os.path.join(args.input_path, args.domain_1, "items.txt")
    item_1 = load_filtered_units(items_path_1)

    items_path_2 = os.path.join(args.input_path, args.domain_2, "items.txt")
    item_2 = load_filtered_units(items_path_2)

    index2item_1 = index2unit(item_1)
    index2item_2 = index2unit(item_2)

    args.item2index_1 = unit2index(index2item_1)
    args.item2index_2 = unit2index(index2item_2)

    # output index2item files
    print("Writing out processed items list... ", end="")

    write_out_items(index2item_1, args.output_path_1)
    write_out_items(index2item_2, args.output_path_2)

    print("Done.")


def split_sets(inters, user2index, item2index, neg_candidates):
    user2pos_items = collections.defaultdict(list)
    user2neg_items = collections.defaultdict(list)
    for inter in inters:
        user, item, rating, _ = inter.split(',')
        rating = int(rating)
        if rating == 1:
            user2pos_items[user2index[user]].append(item2index[item])
        else:
            user2neg_items[user2index[user]].append(item2index[item])

    train, valid, test = [], [], []
    item_set = set([i for i in range(len(item2index))])

    for i in tqdm(range(len(user2index))):      
        pos_list = random.sample(user2pos_items[i], len(user2pos_items[i]))

        non_inter_set = item_set - set(user2pos_items[i] + user2neg_items[i])
        non_inter_list = list(non_inter_set)
        non_inter_valid = random.sample(non_inter_list, neg_candidates)
        non_inter_test = random.sample(non_inter_list, neg_candidates)

        valid.append([i, pos_list[-2]] + non_inter_valid)
        test.append([i, pos_list[-1]] + non_inter_test)
        train.extend([[i, pos, 1] for pos in pos_list[:-2]])
        train.extend([[i, neg, 0] for neg in user2neg_items[i]])

    return train, valid, test


def write_out_inters(path, train, valid, test):
    with open(os.path.join(path, "train.txt"), 'w') as fp:
        fp.write(f"{len(train)}\n")
        for user, item, rating in train:
            fp.write(f"{user}\t{item}\t{rating}\n")
    
    with open(os.path.join(path, "valid.txt"), 'w') as fp:
        fp.write(f"{len(valid)}\n")
        for inter in valid:
            fp.write('\t'.join([str(ele) for ele in inter]) + '\n')

    with open(os.path.join(path, "test.txt"), 'w') as fp:
        fp.write(f"{len(test)}\n")
        for inter in test:
            fp.write('\t'.join([str(ele) for ele in inter]) + '\n')


def split_inters(args):
    # load filtered items in dual domains
    print("Spliting Tr/Va/Te sets...")

    inters_path_1 = os.path.join(args.input_path, args.domain_1, "ratings.csv")
    inters_1 = load_filtered_units(inters_path_1)
    tr_1, va_1, te_1 = split_sets(inters_1, args.user2index_1, args.item2index_1, args.neg_candidates)
    write_out_inters(args.output_path_1, tr_1, va_1, te_1)
    print(f"  Statistics - {args.domain_1}: Train: {len(tr_1)}, Valid: {len(va_1)}, Test: {len(te_1)}")

    inters_path_2 = os.path.join(args.input_path, args.domain_2, "ratings.csv")
    inters_2 = load_filtered_units(inters_path_2)
    tr_2, va_2, te_2 = split_sets(inters_2, args.user2index_2, args.item2index_2, args.neg_candidates)
    write_out_inters(args.output_path_2, tr_2, va_2, te_2)
    print(f"  Statistics - {args.domain_2}: Train: {len(tr_2)}, Valid: {len(va_2)}, Test: {len(te_2)}")

    print("Done.")


domain_fullname_dict = {
    'Amazon': {
        'Music': 'CDs_and_Vinyl',
        'Movie': 'Movies_and_TV',
        'Cell': 'Cell_Phones_and_Accessories',
        'Elec': 'Electronics'       
    }
}


# Set seed
    
def set_seed(seed):
    random.seed(seed)