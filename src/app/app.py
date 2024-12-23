import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from src.app.utils.data import generate_data
from src.app.utils.model import train_model, predict_targets


class App:
    def __init__(self, config_path: str):
        """
        Initializes the application with the given configuration path.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path

    def run(self):
        """
        Runs the Streamlit application.
        """
        st.title("Multi-Output Regression Application with Clusters")

        # Generate and display data (if data not loaded in streamit)
        # if not st.session_state.get("training_data_generated", False):
        X_train, Y_train, category_train = generate_data(n_samples=200)
        st.session_state["training_data_generated"] = True

        model = train_model(X_train, Y_train)

        st.write("# 1. Training data")
        st.write("## 1.1. Training data overview")
        self.display_data_table(X_train, Y_train, category_train, "Training")

        st.write("## 1.2. Descriptive statistics")
        st.write("### Clustering data by category")
        self.display_clusters(X_train, category_train)

        # Show pairplots
        st.write("### Pairplots of the data")
        self.display_pairplots(X_train, category_train)

        st.write("### Summary of targets")
        self.display_target_summary(Y_train)

        st.write("## 1.3. Train model and get results")
        st.write("### Performance on training data")
        self.display_training_performance(model, X_train, Y_train)

        st.write("# 2. Predictions on new data")
        X_new, Y_new, category_new = generate_data(n_samples=10)
        self.display_data_table(X_new, Y_new, category_new, "New")
        self.handle_predictions(model, X_new)

    @staticmethod
    def display_pairplots(X_train, category):
        """
        Displays pairplots of the training data.

        Args:
            X_train (ndarray): Training data.
            category (list): List of categorical labels.
        """

        data = pd.DataFrame(X_train[:, :-3], columns=[f"Feature_{i+1}" for i in range(10)])
        data["Category"] = category
        sns.pairplot(data, hue="Category", palette="Set2")
        st.pyplot(plt)

    @staticmethod
    def display_clusters(X_train, category):
        """
        Displays the PCA visualization of clusters.

        Args:
            X_train (ndarray): Training data.
            category (list): List of categorical labels.
        """
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train[:, :-3].astype(float))  # Exclude encoded columns
        cluster_df = pd.DataFrame({
            "PCA1": X_pca[:, 0],
            "PCA2": X_pca[:, 1],
            "Category": category
        })

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=cluster_df, x="PCA1", y="PCA2", hue="Category", palette="Set2", s=100)
        plt.title("PCA Visualization of Data by Category")
        st.pyplot(plt)

    @staticmethod
    def display_target_summary(Y_train):
        """
        Displays a summary of the target variables.

        Args:
            Y_train (ndarray): Training target data.
        """
        means = Y_train.mean(axis=0)
        stds = Y_train.std(axis=0)
        targets_summary = pd.DataFrame({
            "Target": [f"Target_{i+1}" for i in range(Y_train.shape[1])],
            "Mean": means,
            "Standard Deviation": stds
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(targets_summary["Target"], targets_summary["Mean"], yerr=targets_summary["Standard Deviation"], capsize=5)
        ax.set_title("Mean and Standard Deviation of Targets")
        st.pyplot(fig)

        st.write("Targets Table:")
        st.table(targets_summary)

    @staticmethod
    def display_data_table(X_train, Y_train, category, table_name=""):
        """
        Displays a combined table of features, targets, and categories.

        Args:
            X_train (ndarray): Training data (features and encoded categories).
            Y_train (ndarray): Training target data.
            category (list): List of categorical labels.
        """
        # Extract feature names (excluding encoded category columns)
        feature_names = [f"Feature_{i+1}" for i in range(X_train.shape[1] - len(np.unique(category)))]
        target_names = [f"Target_{i+1}" for i in range(Y_train.shape[1])]

        # Create DataFrame for features
        feature_df = pd.DataFrame(X_train[:, :-len(np.unique(category))], columns=feature_names)

        # Add category and targets
        feature_df["Category"] = category
        target_df = pd.DataFrame(Y_train, columns=target_names)

        # Combine features and targets
        combined_df = pd.concat([feature_df, target_df], axis=1)

        # Display in Streamlit
        st.write(f"### {table_name} Data Table")
        st.dataframe(combined_df)

    @staticmethod
    def display_training_performance(model, X_train, Y_train):
        """
        Displays the training performance metrics.

        Args:
            model (RandomForestRegressor): Trained regression model.
            X_train (ndarray): Training input data.
            Y_train (ndarray): Training target data.
        """
        Y_train_pred = model.predict(X_train)
        mse_train = mean_squared_error(Y_train, Y_train_pred, multioutput='raw_values')
        mse_df = pd.DataFrame(mse_train, columns=["MSE"], index=[f"Target_{i+1}" for i in range(Y_train.shape[1])])
        st.table(mse_df)
        st.write("Average MSE:", mse_train.mean())

    @staticmethod
    def handle_predictions(model, X_new):
        """
        Handles predictions on new data.

        Args:
            model (RandomForestRegressor): Trained regression model.
        """
        if st.button("Predict targets"):
            predictions = predict_targets(model, X_new)
            st.write("Prediction results:")
            st.dataframe(pd.DataFrame(predictions, columns=[f"Target_{i+1}" for i in range(predictions.shape[1])]))