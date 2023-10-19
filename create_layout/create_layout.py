import argparse
import os

import pandas as pd
import numpy as np

from src.app import is_integer_series, is_string_series



def main():
    parser = argparse.ArgumentParser(description="Create a file with the layout of the data")
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--output_path', type=str, default="layout.txt", help='Path to the output file')
    parser.add_argument('--id_cols', type=str, nargs='+', help='Columns that should be treated as IDs')
    parser.add_argument('--delimiter', type=str, default=",", help='Delimiter of the data file')
    args = parser.parse_args()


    # Read the input dataset
    df = pd.read_csv(args.data_path, dtype='str', delimiter=args.delimiter)

    # Initialize a dictionary to store the layout
    layout = {}

    # Process the ID columns. All treated as strings regardless of content.
    for col in args.id_cols:
        # Calculate the string length of the max value in the column
        max_length = df[col].apply(lambda x: len(str(x))).max()
        layout[col] = f'VARCHAR({max_length})'

    # Process the non-ID columns.
    non_ids = [col for col in df.columns if col not in args.id_cols]
    for col in non_ids:

        # Check if the column is a string and cannot be converted to a number
        if is_string_series(df[col]):
            max_length = df[col].apply(lambda x: len(str(x))).max()
            layout[col] = f'VARCHAR({max_length})'
        
        # Check if the column is numeric
        else:

            # Check if the column is an integer, assign SMALLINT, INTEGER, or BIGINT
            if is_integer_series(df[col]):
                length = len(str(df[col].astype(float).max().astype(int)))
                if length < 6:
                    layout[col] = 'SMALLINT'
                elif length < 11:
                    layout[col] = 'INTEGER'
                else:
                    layout[col] = 'BIGINT'

            # Check if the column is a float, assign DECIMAL
            else:
                # Calculate the max number of digits before the decimal
                max_digits = df[col].apply(lambda x: len(str(x).split('.')[0])).max()
                # Calculate the max number of digits after the decimal
                max_decimal = df[col].apply(lambda x: len(str(x).split('.')[1])).max()
                layout[col] = f'DECIMAL({max_digits}, {max_decimal})'


    # If the output file already exists, fail instead of delete since arbitrary 
    # output location could have been entered
    if os.path.exists(args.output_path):
        raise ValueError(f'Output file already exists: {args.output_path}')

    # Write the layout to the output file
    with open(args.output_path, 'w') as f:
        for col, col_type in layout.items():
            f.write(f'{col} {col_type}\n')
        

if __name__ == '__main__':
    main()