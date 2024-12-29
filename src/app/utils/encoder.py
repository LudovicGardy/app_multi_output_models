import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from src.app.utils.data import Data


class CategoryEncoder:
    def __init__(self):
        self.encoder = None
        self.columns = None

    def fit(self, X, category_values):
        """
        Fit the encoder on training data.

        Args:
            X (ndarray): Input feature matrix.
            category_values (List[str]): List of categorical values.
        """
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Category"] = category_values
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(df[["Category"]])
        self.columns = df.columns.tolist()

    def transform(self, X, category_values):
        """
        Transform data based on the fitted encoder.

        Args:
            X (ndarray): Input feature matrix.
            category_values (List[str]): List of categorical values.

        Returns:
            ndarray: Transformed feature matrix with encoded categorical variables.
        """
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Category"] = category_values
        encoded = self.encoder.transform(df[["Category"]])
        df_encoded = pd.concat([df, pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(["Category"]))], axis=1)
        return df_encoded.drop(columns=["Category"]).values


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