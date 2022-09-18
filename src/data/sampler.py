from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import resample


class Sampler:
    def __init__(
        self,
        df: pd.DataFrame,
        min_n_samples: int,
        max_n_samples: int,
        class_col: str,
        replace: bool      = False
    ): 
        self.df            = df
        self.class_col     = class_col
        self.min_n_samples = min_n_samples
        self.max_n_samples = max_n_samples
        self.replace       = replace

    def __call__(self):
        classes = np.unique(self.df[self.class_col].values)
        result = pd.DataFrame()
        for clazz in tqdm (classes, desc="Sampling...", ascii=False):
            rows = self.df[self.df[self.class_col] == clazz]
            if rows.shape[0] < self.min_n_samples:
                continue

            if rows.shape[0] > self.max_n_samples:                
                sample = resample(
                    rows,
                    replace      = self.replace,
                    n_samples    = self.max_n_samples
                )
            else:
                sample = rows
            
            if result.shape[0] >= 0:
                result = result.append(sample)
            else:
                result = sample

        return result