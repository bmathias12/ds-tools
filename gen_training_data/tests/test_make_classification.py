import pytest
from src.make_classification import MakeClassification

def test_generate():
    # Test that generate() method generates the correct number of samples
    mc = MakeClassification(output_dir='.', output_name='test', seed=42)
    mc.generate(n_samples=100, n_features=10, n_informative=5, n_redundant=2, n_repeated=1, n_classes=3)
    assert mc.df.shape == (100, 11)

def test_split():
    # Test that split() method splits the dataset into train, val, and test sets correctly
    mc = MakeClassification(output_dir='.', output_name='test', seed=42)
    mc.generate(n_samples=100, n_features=10, n_informative=5, n_redundant=2, n_repeated=1, n_classes=3)
    mc.split(n_sets=3)
    assert mc.df_train.shape == (80, 11)
    assert mc.df_val.shape == (10, 11)
    assert mc.df_test.shape == (10, 11)

def test_split_raises_error():
    # Test that split() method raises an error if generate() has not been called first
    mc = MakeClassification(output_dir='.', output_name='test', seed=42)
    with pytest.raises(ValueError):
        mc.split(n_sets=3)