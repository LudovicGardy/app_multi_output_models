import numpy as np

from sklearn.ensemble import RandomForestRegressor

from src.app.utils.data import Data
from src.app.utils.encoder import CategoryEncoder, encode_data


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

