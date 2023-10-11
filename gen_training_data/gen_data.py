import argparse
import tempfile

import pandas as pd
from sklearn.datasets import make_classification

def main(num_samples):
    print(num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data')
    parser.add_argument('-n', '--num_samples', type=int, default=1000, help='Number of samples to generate')
    args = parser.parse_args()

    main(args.num_samples)
