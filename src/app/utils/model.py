import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

def train_model(X_encoded: np.ndarray, Y: np.ndarray) -> RandomForestRegressor:
    """
    Trains a Random Forest regressor on the given encoded data.

    Args:
        X_encoded (np.ndarray): Encoded feature matrix.
        Y (np.ndarray): Target data.

    Returns:
        RandomForestRegressor: A trained RandomForestRegressor.
    """
    # Entraînement du modèle
    model = RandomForestRegressor(random_state=42)
    model.fit(X_encoded, Y)
    
    return model

def predict_targets(model: RandomForestRegressor, encoder: CategoryEncoder, data: Data) -> np.ndarray:
    """
    Predicts target values for new input data.

    Args:
        model (RandomForestRegressor): Trained regression model.
        encoder (CategoryEncoder): Fitted CategoryEncoder.
        data (Data): New data object.

    Returns:
        np.ndarray: Predicted target values.
    """
    X_encoded = encode_data(data, encoder)
    return model.predict(X_encoded)

