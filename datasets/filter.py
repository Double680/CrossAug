import os
import argparse
from utils import *


def init():
    # basic configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='Movie', type=str, help='Movie/Music/Cell/Elec')
    parser.add_argument('--user_k', type=int, default=5, help='User k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='Item k-core filtering')
    
    args = parser.parse_args()

    args.dataset = "Amazon"
    assert args.domain in ["Movie", "Music", "Cell", "Elec"]

    print("================ Data Filtering ================")
    print(f"Dataset: {args.dataset}.  Domain: {args.domain}.")

    print("Initializing...")

    # check input/output dir
    input_path = os.path.join(
        "./raw", args.dataset, args.domain
    )
    check_path(input_path)
    output_path = os.path.join(
        "./filtered", args.dataset, args.domain
    )
    check_path(output_path)

    args.input_path = input_path
    args.output_path = output_path

    args.domain_fullname = domain_fullname_dict[args.dataset][args.domain]

    print("Initialized!")

    return args


# Load and filter ratings and metafile

def filter(args):
    print("Filtering...")

    # load ratings file
    rating_inters = load_ratings(args)

    # load metadata file
    meta_items = load_metadata(args)

    # filter records (k-core)
    filter_inters(args, rating_inters, meta_items)

    print("Filtered!")


def write_out(args):
    print("Write out filtered files...")

    # ratings
    ratings_output_path = os.path.join(args.output_path, 'ratings.csv')
    inters_len = len(args.rating_inters)
    with open(ratings_output_path, 'w') as fp:
        fp.write("user,item,rating,timestamp\n")
        for user, item, rating, timestamp in args.rating_inters:
            fp.write(f"{user},{item},{rating},{timestamp}\n")

    # users
    users_path = os.path.join(args.output_path, 'users.txt')
    user_len = len(args.users)
    with open(users_path, 'w') as fp:
        fp.write(f"{user_len}\n")
        users = "\n".join(args.users)
        fp.write(users)

    # items
    items_path = os.path.join(args.output_path, 'items.txt')
    item_len = len(args.items)
    with open(items_path, 'w') as fp:
        fp.write(f"{item_len}\n")
        items = "\n".join(args.items)
        fp.write(items)

    # statistics
    density = 100 * inters_len / (user_len * item_len)
    print("Statistics:")
    print(f"  # Users: {user_len}\n  # Items: {item_len}\n  # Inters: {inters_len}\n  Density: {density:.4f}%")

    print("Success!")


if __name__ == '__main__':
    args = init()
    filter(args)
    write_out(args)