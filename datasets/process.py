import os
import argparse
from utils import *


def init():
    # basic configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', default='Movie-Music', help='Movie-Music/Cell-Elec', type=str)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--neg_candidates', default=499, type=int)
    
    args = parser.parse_args()
    
    args.dataset = "Amazon"
    assert args.domains in ["Cell-Elec", "Movie-Music"] 
    domains = args.domains.split("-")
    args.domain_1, args.domain_2 = domains[0], domains[1]

    print("================ Data Processing ================")
    print(f"Dataset: {args.dataset}.  Domain 1: {args.domain_1}.  Domain 2: {args.domain_2}.")

    print("Initializing...")

    # set seed
    set_seed(args.seed)

    # check input/output dir
    input_path = os.path.join(
        "./filtered", args.dataset
    )
    check_path(input_path)

    output_path = os.path.join(
        "./processed", args.dataset, f"{args.domain_1}-{args.domain_2}"
    )
    check_path(output_path)

    output_path_1 = os.path.join(output_path, args.domain_1)
    check_path(output_path_1)

    output_path_2 = os.path.join(output_path, args.domain_2)
    check_path(output_path_2)

    args.input_path = input_path
    args.output_path_1 = output_path_1
    args.output_path_2 = output_path_2

    print("Initialized!")

    return args


# Load and process dual-target domain data

def process(args):
    print("Processing...")

    # summarize statistics of union/joint/domain-specific users
    process_users(args)
    
    # summarize statistics of items
    process_items(args)

    # split Train/Valid/Test set from interactions
    split_inters(args)

    print("Processed!")


if __name__ == '__main__':
    args = init()
    process(args)