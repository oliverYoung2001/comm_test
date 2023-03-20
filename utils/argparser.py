import argparse
import datetime as date
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus",
                        type=int,
                        default=1,
                        help="""Number of GPUs with which to run inference""")
    
    args = parser.parse_args()
    return args