import numpy as np

from sklearn.ensemble import RandomForestRegressor

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

def predict_targets(model: RandomForestRegressor, X_encoded: np.ndarray) -> np.ndarray:
    """
    Predicts target values for new input data.

    Args:
        model (RandomForestRegressor): Trained regression model.
        encoder (CategoryEncoder): Fitted CategoryEncoder.
        data (Data): New data object.

    Returns:
        np.ndarray: Predicted target values.
    """
    return model.predict(X_encoded)

