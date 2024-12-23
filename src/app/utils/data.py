import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from typing import Tuple

def generate_data(n_samples, categories=None) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generates training or prediction data with explicit categories for encoding.

    Args:
        n_samples (int): Number of samples to generate.
        categories (list, optional): List of all possible categories for encoding.

    Returns:
        Tuple[np.ndarray, np.ndarray, list]: Input data, target data, and categories.
    """
    X_continuous, Y_train = make_regression(n_samples=n_samples, n_features=10, n_targets=5, noise=0.1, random_state=42)
    category = np.random.choice(categories if categories else ["Level 1", "Level 2", "Level 3"], size=n_samples)

    # Add bias
    bias = {
        "Level 1": 2,
        "Level 2": -2,
        "Level 3": 0
    }
    category_bias = np.array([bias[cat] for cat in category])
    X_continuous[:, 0] += category_bias

    # Encode categorical feature with fixed columns
    if not categories:
        categories = ["Level 1", "Level 2", "Level 3"]

    category_encoded = pd.get_dummies(category, drop_first=False).reindex(columns=categories, fill_value=0).values
    X_train_encoded = np.hstack((X_continuous, category_encoded))

    return X_train_encoded, Y_train, category
