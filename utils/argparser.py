import argparse
import datetime as date
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus",
                        type=int,
                        default=1,
                        help="""Number of GPUs with which to run inference""")
    
    parser.add_argument("--gpuids",
                        type=str,
                        default='',
                        help="""GPUIDs for p2p comm between nodes""")
    
    parser.add_argument("--excel_file",
                        type=str,
                        default='',
                        help="""file name of excel""")
    args = parser.parse_args()
    return args