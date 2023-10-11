import argparse
import tempfile

import pandas as pd
from sklearn.datasets import make_classification

def main(num_samples, num_features):
    
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
    )

    df_X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')

    print(df_X.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data')
    parser.add_argument('-n', '--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('-f', '--num_features', type=int, default=20, help='Number of features to generate')
    args = parser.parse_args()

    main(args.num_samples, args.num_features)
