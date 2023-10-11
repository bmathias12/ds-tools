import argparse
import os
import tempfile


import pandas as pd
from sklearn.datasets import make_classification

def main(output_dir, output_name, n_sets, n_samples, n_features, seed,):
    
    make_classification = MakeClassification(
        output_dir=output_dir,
        output_name=output_name,
        n_sets=n_sets,
        seed=seed,
    )

    make_classification.generate(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
    )

    out_loc = os.path.join(output_dir, output_name)
    print(out_loc)

class MakeClassification:
    def __init__(self, output_dir, output_name, n_sets, seed):
        self.output_dir = output_dir
        self.output_name = output_name
        self.n_sets = n_sets
        self.seed = seed

    def generate(self, n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes):

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            random_state=self.seed,
        )
        df_X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y, name='target')

        self.df = pd.concat([df_X, y], axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data')

    parser.add_argument('-o', '--output_dir', type=str, default=tempfile.gettempdir(), help='Output directory')
    parser.add_argument('--output_name', type=str, default='gen_ml_data', help='Name of the dataset')
    parser.add_argument('--n_sets', type=int, choices=[1, 2, 3], help='Choose 1, 2, or 3 for the number of sets to generate')
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('-f', '--n_features', type=int, default=20, help='Number of features to generate')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()

    main(args.output_dir, args.output_name, args.n_sets, args.n_samples, args.n_features, args.seed)
