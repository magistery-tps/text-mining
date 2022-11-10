import numpy as np
import torch
import data as dt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_col, target_col, tokenizer):
        self.features = [tokenizer.tokenize(feat) for feat in df[feature_col]]
        self.target = df[target_col].values

    @property
    def shape(self): return self.target.shape

    def __len__(self): return len(self.target)

    def __getitem__(self, idx): return self.features[idx], self.target[idx]