"""
Uses scikit-learn's make_classification to generate training data.

Args:
    output_dir (str): Output directory
    output_name (str): Name of the dataset (will be prefix of the output files)
    n_sets (int): Number of sets to generate (1, 2, or 3, for train, val, and test sets)
    n_samples (int): Number of samples to generate
    n_features (int): Number of features to generate
    seed (int): Random seed

Usage:
    python gen_data.py -o /path/to/output_dir --output_name gen_ml_data --n_sets 1 -n 1000 -f 20 -s 123

"""
import argparse
import os
import tempfile

from src.make_classification import MakeClassification


def main(output_dir, output_name, n_sets, n_samples, n_features, seed,):
    
    make_classification = MakeClassification(
        output_dir=output_dir,
        output_name=output_name,
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

    make_classification.split(n_sets=n_sets)

    # Write the sets to disk
    make_classification.write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data')

    parser.add_argument('-o', '--output_dir', type=str, default=tempfile.gettempdir(), help='Output directory')
    parser.add_argument('--output_name', type=str, default='gen_ml_data', help='Name of the dataset')
    parser.add_argument('--n_sets', type=int, choices=[1, 2, 3], default=1, help='Choose 1, 2, or 3 for the number of sets to generate')
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('-f', '--n_features', type=int, default=20, help='Number of features to generate')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()

    main(args.output_dir, args.output_name, args.n_sets, args.n_samples, args.n_features, args.seed)
