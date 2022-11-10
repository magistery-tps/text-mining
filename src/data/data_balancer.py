from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import resample


class DataBalancer:
    """
    * Usefull to perform dataset balancing.
    * Exclude instances for classes with less that min_n_instances.
    * Sample a max_n_instances count for classes with more than max_n_instances.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        min_n_instances: int,
        max_n_instances: int,
        class_col: str,
        replace: bool      = False
    ):
        """
        Args:
            df (pd.DataFrame): dataframe when perform instances sampling
            min_n_instances (int): Min instances count class. Classes with las than min_n_instances are skiped.
            max_n_instances (int): Max instances counhis It Sampler sample(max_n) instances for clases with more than max_n_instances.
            class_col (str): dataframe column that represent a class.
            replace (bool, optional): Sample wout or witout replacement. Defaults to False.
        """
        self.df            = df
        self.class_col     = class_col
        self.min_n_instances = min_n_instances
        self.max_n_instances   = max_n_instances
        self.replace       = replace

    def __call__(self):
        classes = np.unique(self.df[self.class_col].values)
        result = pd.DataFrame()
        for clazz in tqdm (classes, desc="Sampling...", ascii=False):
            rows = self.df[self.df[self.class_col] == clazz]
            if rows.shape[0] < self.min_n_instances:
                continue

            if rows.shape[0] > self.max_n_instances:
                sample = resample(
                    rows,
                    replace      = self.replace,
                    n_samples    = self.max_n_instances
                )
            else:
                sample = rows

            if result.shape[0] >= 0:
                result = result.append(sample)
            else:
                result = sample

        return result