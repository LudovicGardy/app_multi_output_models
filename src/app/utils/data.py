import numpy as np
import pandas as pd
from typing import List

from dataclasses import dataclass, field

@dataclass
class Data:
    X: np.ndarray = field(default_factory=lambda: np.array([]))
    Y: np.ndarray = field(default_factory=lambda: np.array([]))
    category: List[str] = field(default_factory=list)
    category_columns: List[str] = field(default_factory=list)

    @staticmethod
    def load_from_file(file):
        df = pd.read_csv(file, sep=",")

        target_columns = [col for col in df.columns if 'target' in col.lower()]
        feature_columns = [col for col in df.columns if col not in target_columns + ['Category']]

        X = df[feature_columns].values
        Y = df[target_columns].values
        category_columns = ['Category']
        category = df['Category'].astype(str).tolist()

        return Data(X, Y, category, category_columns=category_columns)
