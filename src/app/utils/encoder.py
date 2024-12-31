import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from typing import Dict

from src.app.utils.data import Data


class CategoryEncoder:
    def __init__(self):
        self.encoder = None
        self.columns = None

    def fit(self, X: np.ndarray, categories: Dict[str, list]):
        """
        Fit the encoder on training data.

        Args:
            X (ndarray): Input feature matrix.
            categories (Dict[str, list]): Dictionary with multiple category columns.
        """
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])

        # Vérifier la cohérence des longueurs
        for key, values in categories.items():
            if len(values) != len(X):
                raise ValueError(f"Length of category '{key}' does not match the number of rows in X.")

        # Ajouter les colonnes catégorielles
        for key, values in categories.items():
            df[key] = values

        # Stocker les noms des colonnes catégorielles
        self.categorical_columns = list(categories.keys())

        # Fit the OneHotEncoder
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(df[self.categorical_columns])

    def transform(self, X: np.ndarray, categories: Dict[str, list]):
        """
        Transform data based on the fitted encoder.

        Args:
            X (ndarray): Input feature matrix.
            categories (Dict[str, list]): Dictionary of categorical values.

        Returns:
            ndarray: Transformed feature matrix with encoded categorical variables.
        """
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])

        # Vérifier la cohérence des longueurs
        for key, values in categories.items():
            if len(values) != len(X):
                raise ValueError(f"Length of category '{key}' does not match the number of rows in X.")

        # Ajouter les colonnes catégorielles
        for key, values in categories.items():
            df[key] = values

        # Vérifier que les colonnes correspondent à celles utilisées lors du fit
        if set(self.categorical_columns) != set(categories.keys()):
            raise ValueError("Categories provided in transform do not match those used in fit.")

        # Transformer les colonnes catégorielles
        encoded = self.encoder.transform(df[self.categorical_columns])
        encoded_columns = self.encoder.get_feature_names_out(self.categorical_columns)

        # Créer un DataFrame concaténé
        df_encoded = pd.concat([df, pd.DataFrame(encoded, columns=encoded_columns, index=df.index)], axis=1)

        # Supprimer les colonnes catégorielles originales
        return df_encoded.drop(columns=self.categorical_columns).values

def encode_data(data: Data, encoder: CategoryEncoder) -> np.ndarray:
    """
    Encodes the input data using the provided encoder.

    Args:
        data (Data): Data object containing features and categories.
        encoder (CategoryEncoder): Fitted CategoryEncoder.

    Returns:
        np.ndarray: Encoded feature matrix.
    """
    return encoder.transform(data.X, data.categories)