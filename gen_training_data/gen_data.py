import argparse
import os
import tempfile


import pandas as pd
from sklearn.datasets import make_classification

def main(output_dir, output_name, n_sets, num_samples, num_features, seed,):
    
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
    )

    df_X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')

    out_loc = os.path.join(output_dir, output_name)
    print(out_loc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data')

    parser.add_argument('-o', '--output_dir', type=str, default=tempfile.gettempdir(), help='Output directory')
    parser.add_argument('--output_name', type=str, default='gen_ml_data', help='Name of the dataset')
    parser.add_argument('--n_sets', type=int, choices=[1, 2, 3], help='Choose 1, 2, or 3 for the number of sets to generate')
    parser.add_argument('-n', '--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('-f', '--num_features', type=int, default=20, help='Number of features to generate')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()

    main(args.output_dir, args.output_name, args.n_sets, args.num_samples, args.num_features, args.seed)
