import numpy as np
import pandas as pd
from typing import List, Dict
import streamlit as st

from dataclasses import dataclass, field

from src.app.utils.encoder import CategoryEncoder

@dataclass
class Data:
    all_columns: List[str] = field(default_factory=list)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_encoded: pd.DataFrame = field(default_factory=pd.DataFrame)
    X: np.ndarray = field(default_factory=lambda: np.array([]))
    Y: np.ndarray = field(default_factory=lambda: np.array([]))
    X_labels: List[str] = field(default_factory=list)
    Y_labels: List[str] = field(default_factory=list)

    def load_from_file(self, file):
        """
        Load data from a CSV file and automatically detect categorical columns.
        
        Args:
            file (str): Path to the CSV file.
            
        Returns:
            Data: An instance of the Data class containing the data.
        """
        # Load the CSV file
        self.df = pd.read_csv(file, sep=",")

        # Identify all columns
        self.all_columns = self.df.columns.tolist()

    def get_feature_and_target_df(self, feature_columns, target_columns):
        self.features_df = self.df_encoded[feature_columns]
        self.target_df = self.df_encoded[target_columns]

    def encode_categorical_columns(self, category_columns):
        """
        Encode categorical columns using One-Hot Encoding.
        
        Args:
            category_columns (list): List of categorical column names.
        """
        encoder = CategoryEncoder(encoding_type="label")
        self.df_encoded = encoder.fit_transform(self.df, category_columns)