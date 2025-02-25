import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train_model(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def predict_targets(self, model, X: np.ndarray) -> np.ndarray:
        pass


class RandomForestModel(BaseModel):
    def train_model(self, X_encoded: np.ndarray, Y: np.ndarray) -> RandomForestRegressor:
        model = RandomForestRegressor(random_state=42)
        model.fit(X_encoded, Y)
        return model

    def predict_targets(self, model: RandomForestRegressor, X_encoded: np.ndarray) -> np.ndarray:
        return model.predict(X_encoded)


class CatBoostNativeModel(BaseModel):
    def train_model(self, X, Y, cat_features: list = []) -> CatBoostRegressor:
        """
        Train a CatBoost model using the native approach for multi-target regression.
        For CatBoost to handle multiple targets, we use an appropriate loss function (MultiRMSE).
        """
        model = CatBoostRegressor(
            random_state=42,
            verbose=0,
            loss_function='MultiRMSE',
            eval_metric='MultiRMSE'
        )
        model.fit(X, Y, cat_features=cat_features)
        return model

    def predict_targets(self, model: CatBoostRegressor, X) -> np.ndarray:
        return model.predict(X)
    
class CatBoostMultiRegressorModel(BaseModel):
    def train_model(self, X: np.ndarray, Y: np.ndarray, cat_features: list = []) -> MultiOutputRegressor:
        """
        Train a CatBoost model using the MultiOutputRegressor wrapper for multi-target
        regression. This approach is useful when we want to use the same model for multiple
        targets, and it is compatible with the scikit-learn API for model training and prediction.
        The difference with the native approach is that we can use the same model for multiple
        targets without the need to specify a loss function that supports multiple targets.
        In this approach, we train a separate model for each target.
        """
        base_model = CatBoostRegressor(random_state=42, verbose=0)
        model = MultiOutputRegressor(base_model)
        model.fit(X, Y, **{"cat_features": cat_features})
        return model

    def predict_targets(self, model: MultiOutputRegressor, X: np.ndarray) -> np.ndarray:
        return model.predict(X)