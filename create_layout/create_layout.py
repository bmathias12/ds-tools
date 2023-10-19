import argparse
import os

import pandas as pd
import numpy as np



def main():
    parser = argparse.ArgumentParser(description="Create a file with the layout of the data")
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--output_path', type=str, help='Path to the output file')
    args = parser.parse_args()

if __name__ == '__main__':
    main()