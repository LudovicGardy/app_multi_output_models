import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train_model(self, X: np.ndarray, Y: np.ndarray):
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


class CatBoostModel(BaseModel):
    def train_model(self, X: np.ndarray, Y: np.ndarray, cat_features: list = []) -> MultiOutputRegressor:
        base_model = CatBoostRegressor(random_state=42, verbose=0)
        model = MultiOutputRegressor(base_model)
        model.fit(X, Y, **{"cat_features": cat_features})
        return model

    def predict_targets(self, model: MultiOutputRegressor, X: np.ndarray) -> np.ndarray:
        return model.predict(X)
