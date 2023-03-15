#!/usr/bin/env python3

import torch
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="""DIR""",
    )

    parser.add_argument(
        "--data-len",
        type=int,
        default=0,
        help="""DIR""",
    )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    data_dic = torch.load(args.data_dir)
    for k in data_dic.keys():
        data_dic[k] = data_dic[k][:args.data_len]
    torch.save(data_dic, "data/temp.pt")