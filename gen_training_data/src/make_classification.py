import pandas as pd
from sklearn.datasets import make_classification


class MakeClassification:
    """Generate a classification dataset
    
    Attributes:
        output_dir (str): Output directory
        output_name (str): Name of the dataset
        seed (int): Random seed
    """
    def __init__(self, output_dir, output_name, seed):
        self.output_dir = output_dir
        self.output_name = output_name
        self.seed = seed

    def generate(self, n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes):
        """Generate a classification dataset
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features to generate
            n_informative (int): Number of informative features
            n_redundant (int): Number of redundant features
            n_repeated (int): Number of duplicated features
            n_classes (int): Number of classes

        Returns:
            None
        """
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

    def split(self, n_sets):
        """Split the dataset into train, validation, and test sets
        
        Args:
            n_sets (int): Number of sets to generate
        Returns:
            None
        """
        if not hasattr(self, 'df'):
            raise ValueError('Dataset has not been generated yet. Run `generate` method first.')
        
        if n_sets == 1:
            self.df_train = self.df
            self.df_val = None
            self.df_test = None
        elif n_sets == 2:
            self.df_train = self.df.sample(frac=0.8, random_state=self.seed)
            self.df_val = self.df.drop(self.df_train.index)
            self.df_test = None
        elif n_sets == 3:
            self.df_train = self.df.sample(frac=0.8, random_state=self.seed)
            self.df_val = self.df.drop(self.df_train.index).sample(frac=0.5, random_state=self.seed)
            self.df_test = self.df.drop(self.df_train.index).drop(self.df_val.index)
        else:
            raise ValueError('n_sets must be 1, 2, or 3.')
        
    def write(self):
        """Write the train, validation, and test sets to disk
        
        Args:
            None
        Returns:
            None
        """
        if not hasattr(self, 'df_train'):
            raise ValueError('Dataset has not been split yet. Run `split` method first.')
        
        self.df_train.to_csv(f'{self.output_dir}/{self.output_name}_train.csv', index=False)
        if self.df_val is not None:
            self.df_val.to_csv(f'{self.output_dir}/{self.output_name}_val.csv', index=False)
        if self.df_test is not None:
            self.df_test.to_csv(f'{self.output_dir}/{self.output_name}_test.csv', index=False)