import pandas as pd


def load_datasets(path, prefix=''):
    train_set = pd.read_csv(f'{path}/{prefix}train.csv').drop_duplicates()
    val_set   = pd.read_csv(f'{path}/{prefix}val.csv').drop_duplicates()
    test_set  = pd.read_csv(f'{path}/{prefix}test.csv').drop_duplicates()

    train_set['features'] = train_set['features'].apply(str)
    val_set  ['features'] = val_set  ['features'].apply(str)
    test_set ['features'] = test_set ['features'].apply(str)
    
    return train_set, val_set, test_set