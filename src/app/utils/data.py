import numpy as np
import pandas as pd
from typing import List, Dict

from dataclasses import dataclass, field


@dataclass
class Data:
    X: np.ndarray = field(default_factory=lambda: np.array([]))
    Y: np.ndarray = field(default_factory=lambda: np.array([]))
    categories: Dict[str, List[str]] = field(default_factory=dict)
    category_columns: List[str] = field(default_factory=list)
    all_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)

    @staticmethod
    def load_from_file(file):
        """
        Load data from a CSV file and automatically detect categorical columns.
        
        Args:
            file (str): Path to the CSV file.
            
        Returns:
            Data: An instance of the Data class containing the data.
        """
        # Load the CSV file
        df = pd.read_csv(file, sep=",")

        # Identify all columns
        all_columns = df.columns.tolist()

        # Identify categorical columns
        category_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Convert categorical columns to a list of values for each category
        categories = {col: df[col].astype(str).tolist() for col in category_columns}

        # Identify target columns (those containing "target" in their name, case insensitive)
        target_columns = [col for col in df.columns if 'target' in col.lower()]

        # Identify feature columns
        feature_columns = [col for col in df.columns if col not in target_columns + category_columns]

        # Extract feature and target values
        X = df[feature_columns].values
        Y = df[target_columns].values

        return Data(
            X, Y, categories, category_columns, all_columns,
            feature_columns, target_columns
        )