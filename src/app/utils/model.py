from sklearn.ensemble import RandomForestRegressor
import numpy as np
from src.app.utils.data import Data

def train_model(data: Data) -> RandomForestRegressor:
    """
    Trains a Random Forest regressor on the given data.

    Args:
        data (Data): Training data object.

    Returns:
        RandomForestRegressor: Trained regression model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(data.X, data.Y)
    return model

def predict_targets(model: RandomForestRegressor, data: Data) -> np.ndarray:
    """
    Predicts target values for new input data.

    Args:
        model (RandomForestRegressor): Trained regression model.
        data (Data): New data object.

    Returns:
        np.ndarray: Predicted target values.
    """
    return model.predict(data.X)
