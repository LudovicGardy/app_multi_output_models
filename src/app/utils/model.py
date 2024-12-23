from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_model(X_train: np.ndarray, Y_train: np.ndarray) -> RandomForestRegressor:
    """
    Trains a Random Forest regressor on the given data.

    Args:
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training target data.

    Returns:
        RandomForestRegressor: Trained regression model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, Y_train)
    return model

def predict_targets(model: RandomForestRegressor, X_new: np.ndarray) -> np.ndarray:
    """
    Predicts target values for new input data.

    Args:
        model (RandomForestRegressor): Trained regression model.
        X_new (np.ndarray): New input data.

    Returns:
        np.ndarray: Predicted target values.
    """
    return model.predict(X_new)
