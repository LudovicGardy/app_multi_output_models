from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from src.app.utils.data import Data
from src.app.utils.model import RandomForestModel, CatBoostNativeModel, CatBoostMultiRegressorModel


class App:
    def __init__(self, config_path: str, train_data: Data, test_data: Data):
        """
        Initializes the application with the given configuration path.

        Args:
            config_path (str): Path to the configuration file.
            train_data (Data): Training data object.
            test_data (Data): Test data object.
        """
        self.config_path = config_path
        self.train_data = train_data
        self.test_data = test_data

    def run(self):
        """
        Runs the Streamlit application.
        """
        if st.session_state["selected_model"] == "Random Forest":
            model_class = RandomForestModel()
        elif st.session_state["selected_model"] == "CatBoost Native":
            model_class = CatBoostNativeModel()
        elif st.session_state["selected_model"] == "CatBoost MultiRegressor":
            model_class = CatBoostMultiRegressorModel()

        st.session_state["data_loaded"] = True

        tab1, tab2, tab3 = st.tabs(["1️⃣ Training Data", "2️⃣ Model Training", "3️⃣ Predictions On New Data"])

        with tab1:
            with st.expander("Show Training Data Details"):
                st.write("### Training Data")
                st.write("Number of features:", self.train_data.features_df.shape[1])
                st.write("Number of targets:", self.train_data.target_df.shape[1])
                st.write("Number of samples:", self.train_data.features_df.shape[0])

            with st.expander("Show Training Data Table"):
                st.dataframe(st.session_state["train_data"].df)

            with st.expander("Show Encoded Data Table"):
                st.dataframe(st.session_state["train_data"].df_encoded)

            with st.expander("Show Clusters"):
                self.display_clusters()

            with st.expander("Show Pairplots"):
                st.write("### Features")
                self.display_pairplots(
                    st.session_state["train_data"].df_encoded[st.session_state["feature_columns"]],
                    st.session_state["train_data"].df_encoded[st.session_state["clustering_column"]],
                )
                st.write("### Targets")
                self.display_pairplots(
                    st.session_state["train_data"].df_encoded[st.session_state["target_columns"]],
                    st.session_state["train_data"].df_encoded[st.session_state["clustering_column"]],
                )

            with st.expander("Show Targets Summary"):
                self.display_target_summary(
                    st.session_state["train_data"].df_encoded[st.session_state["target_columns"]]
                )

        with tab2:
            st.write("## Train model and get results")
            model = model_class.train_model(
                st.session_state["train_data"].df_encoded[st.session_state["feature_columns"]].values,
                st.session_state["train_data"].df_encoded[st.session_state["target_columns"]].values,
            )
            self.display_training_performance(model, self.train_data)

        with tab3:
            st.write("# Predictions on new data")
            with st.expander("Show Test Data Details"):
                st.write("### Test Data")
                st.write("Number of features:", self.test_data.features_df.shape[1])
                st.write("Number of targets:", self.test_data.target_df.shape[1])
                st.write("Number of samples:", self.test_data.features_df.shape[0])

            with st.expander("Show Test Data Table"):
                st.dataframe(st.session_state["test_data"].df)
            predictions_df = self.handle_predictions(
                model_class, model, st.session_state["test_data"].df_encoded[st.session_state["feature_columns"]].values
            )

            if predictions_df is not None:
                st.write("### Differences between real and predicted values")
                self.create_difference_df(self.test_data.target_df, predictions_df)

    @staticmethod
    def display_clusters():
        """
        Displays the PCA visualization of clusters using Plotly.
        """
        mycat_col = st.session_state["clustering_column"]
        df = st.session_state["train_data"].features_df

        # Split the DataFrame into numerical and categorical columns
        X = df.drop(columns=[mycat_col])
        y = df[mycat_col]

        # Clustering with K-Means
        n_clusters = len(y.unique())
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Traaining on X (the 10 numerical columns)
        clusters = kmeans.fit_predict(X)

        # Add the cluster column to the DataFrame
        df["cluster"] = clusters

        # Reduction of dimensionality for visualization (PCA)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        df["pca1"] = X_pca[:, 0]
        df["pca2"] = X_pca[:, 1]

        # Visualize the clusters
        fig = px.scatter(
            df,
            x="pca1",
            y="pca2",
            color=mycat_col,  # Coloration of the points according to the categorical column
            symbol="cluster",  # We can distinguish the clusters via symbols
            hover_data=["cluster"],  # Display the cluster number when hovering over a point
        )

        fig.update_layout(title="Visualisation PCA - K-Means")

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    def create_difference_df(self, test_df, predictions_df):
        """
        Create a DataFrame that contains the difference between test_df and predictions_df
        for each common column and row, and display the resulting DataFrame and plot.

        Parameters:
        test_df (pd.DataFrame): The test DataFrame.
        predictions_df (pd.DataFrame): The predictions DataFrame.

        Returns:
        pd.DataFrame: The DataFrame containing the differences.
        """
        # Ensure both DataFrames have the same columns
        common_columns = test_df.columns.intersection(predictions_df.columns)

        # Calculate the difference
        difference_df = test_df[common_columns] - predictions_df[common_columns]

        # Display the resulting DataFrame
        st.dataframe(difference_df, width=1000)

        # Plot the differences using Plotly
        fig = px.box(difference_df, title="Differences between Test and Predicted Values")
        fig.update_layout(yaxis_title="Difference")
        st.plotly_chart(fig)

        return difference_df

    @staticmethod
    def display_pairplots(data: np.ndarray, category: list):
        """
        Displays pairplots of the training data.

        Args:
            X_train (ndarray): Training data.
            category (list): List of categorical labels.
        """
        # Generate column names based on the actual number of features
        df = pd.DataFrame(data)
        # Use a non-conflicting name
        df["__pairplot_category__"] = category

        sns.pairplot(df, hue="__pairplot_category__", palette="Set2")
        st.pyplot(plt)

    @staticmethod
    def display_target_summary(Y_train):
        """
        Displays a summary of the target variables using Plotly.

        Args:
            Y_train (ndarray): Training target data.
        """
        means = Y_train.mean(axis=0)
        stds = Y_train.std(axis=0)
        targets_summary = pd.DataFrame({"Target": range(Y_train.shape[1]), "Mean": means, "Standard Deviation": stds})

        fig = px.bar(
            targets_summary,
            x="Target",
            y="Mean",
            error_y="Standard Deviation",
            title="Mean and Standard Deviation of Targets",
            template="plotly_white",
            labels={"Mean": "Mean Value"},
        )
        st.plotly_chart(fig)

        st.dataframe(targets_summary, width=1000)

    @staticmethod
    def display_training_performance(model, data: Data):
        """
        Displays the training performance metrics.

        Args:
            model (RandomForestRegressor): Trained regression model.
            data (Data): Training data object.
        """
        # Ensure the feature columns are correctly specified
        feature_columns = st.session_state["feature_columns"]

        # Predictions on the training set
        Y_train_pred = model.predict(st.session_state["train_data"].df_encoded[feature_columns].values)

        # Calculate the mean squared error (MSE) for each target
        mse_train = mean_squared_error(data.target_df.values, Y_train_pred, multioutput="raw_values")

        # Create a DataFrame for clear display
        mse_df = pd.DataFrame({"MSE": mse_train}, index=[f"Target_{i + 1}" for i in range(data.target_df.shape[1])])

        # Display in Streamlit
        st.write("### Performance on training data")
        st.table(mse_df)
        st.write("MSE mean:", mse_train.mean())

    @staticmethod
    def handle_predictions(model_class, model, X_encoded: np.ndarray) -> Optional[pd.DataFrame]:
        """
        Handles predictions on new data.

        Args:
            model (RandomForestRegressor): Trained regression model.
            X_encoded (np.ndarray): Encoded feature matrix.
        """
        if st.button("Predict targets"):
            predictions = model_class.predict_targets(model, X_encoded)
            st.write("### Prediction Results")
            # Use the target column names from the training data
            target_columns = st.session_state["target_columns"]
            predictions_df = pd.DataFrame(predictions, columns=target_columns)
            st.dataframe(predictions_df, width=1000)
            return predictions_df
        else:
            return None
