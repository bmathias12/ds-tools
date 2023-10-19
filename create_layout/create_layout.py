import argparse
import os

import pandas as pd
import numpy as np



def main():
    parser = argparse.ArgumentParser(description="Create a file with the layout of the data")
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--output_path', type=str, default="layout.txt", help='Path to the output file')
    parser.add_argument('--id_cols', type=str, nargs='+', help='Columns that should be treated as IDs')
    parser.add_argument('--delimiter', type=str, default=",", help='Delimiter of the data file')
    args = parser.parse_args()


    # Read the input dataset
    df = pd.read_csv(args.data_path, dtype='str', delimiter=args.delimiter)

if __name__ == '__main__':
    main()