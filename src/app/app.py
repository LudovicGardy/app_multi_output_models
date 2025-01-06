import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Optional

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from src.app.utils.data import Data
from src.app.utils.model import train_model, predict_targets
from src.app.utils.encoder import CategoryEncoder, encode_data


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
        # Generate and display data (if data not loaded in streamit)
        # if not st.session_state.get("data_loaded", False):
        st.session_state["data_loaded"] = True

        tab1, tab2, tab3 = st.tabs(["1️⃣ Training Data", "2️⃣ Model Training", "3️⃣ Predictions On New Data"])

        with tab1:
            with st.expander("Show Training Data Details"):
                st.write("### Training Data")
                st.write("Number of features:", self.train_data.X.shape[1])
                st.write("Number of targets:", self.train_data.Y.shape[1])
                st.write("Number of samples:", self.train_data.X.shape[0])

            with st.expander("Show Training Data Table"):
                train_df = self.display_data_table(self.train_data.X, self.train_data.Y, self.train_data.categories, "Training")

            with st.expander("Show Clusters"):
                clustering_category = st.session_state["clustering_column"]
                self.display_clusters(self.train_data.X, self.train_data.categories[clustering_category])

            with st.expander("Show Pairplots"):
                st.write("### Features")
                self.display_pairplots(self.train_data.X, self.train_data.categories[clustering_category], "Feature")
                st.write("### Targets")
                self.display_pairplots(self.train_data.Y, self.train_data.categories[clustering_category], "Target")

            with st.expander("Show Targets Summary"):
                self.display_target_summary(self.train_data.Y)

        with tab2:
            st.write("## Train model and get results")
            
            # Fit encoder and encode data
            encoder = CategoryEncoder()
            encoder.fit(self.train_data.X, self.train_data.categories)
            X_encoded = encode_data(self.train_data, encoder)
            
            # Train model
            model = train_model(X_encoded, self.train_data.Y)
            self.display_training_performance(model, encoder, self.train_data)

        with tab3:
            st.write("# Predictions on new data")
            with st.expander("Show Test Data Details"):
                st.write("### Test Data")
                st.write("Number of features:", self.test_data.X.shape[1])
                st.write("Number of targets:", self.test_data.Y.shape[1])
                st.write("Number of samples:", self.test_data.X.shape[0])

            with st.expander("Show Test Data Table"):
                test_df = self.display_data_table(self.test_data.X, self.test_data.Y, self.test_data.categories, "New")
            predictions_df = self.handle_predictions(model, encoder, self.test_data)

            if predictions_df is not None:
                st.write("### Differences between real and predicted values")
                self.create_difference_df(test_df, predictions_df)

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
    def display_pairplots(data: np.ndarray, category: list, parameters: str):
        """
        Displays pairplots of the training data.

        Args:
            X_train (ndarray): Training data.
            category (list): List of categorical labels.
        """
        # Dynamically determine the number of features
        num_parameters = data.shape[1]
        
        # Generate column names based on the actual number of features
        df = pd.DataFrame(data)
        # Use a non-conflicting name
        df["__pairplot_category__"] = category

        sns.pairplot(df, hue="__pairplot_category__", palette="Set2")
        st.pyplot(plt)

    @staticmethod
    def display_clusters(X_train, category):
        """
        Displays the PCA visualization of clusters using Plotly.

        Args:
            X_train (ndarray): Training data.
            category (list): List of categorical labels.
        """
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train[:, :-3].astype(float))  # Exclude encoded columns
        cluster_df = pd.DataFrame({
            "PCA1": X_pca[:, 0],
            "PCA2": X_pca[:, 1],
            "__cluster_category__": category
        })

        fig = px.scatter(
            cluster_df,
            x="PCA1",
            y="PCA2",
            color="__cluster_category__",
            title="PCA Visualization of Data by Category",
            labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"},
        )
        st.plotly_chart(fig)

    @staticmethod
    def display_target_summary(Y_train):
        """
        Displays a summary of the target variables using Plotly.

        Args:
            Y_train (ndarray): Training target data.
        """
        means = Y_train.mean(axis=0)
        stds = Y_train.std(axis=0)
        targets_summary = pd.DataFrame({
            "Target": range(Y_train.shape[1]),
            "Mean": means,
            "Standard Deviation": stds
        })

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
    def display_data_table(X_train, Y_train, categories, table_name="") -> pd.DataFrame:
        """
        Affiche une table combinée des features, des cibles et des catégories.

        Args:
            X_train (ndarray): Données d'entraînement (features).
            Y_train (ndarray): Données cibles d'entraînement.
            categories (dict): Dictionnaire des colonnes catégorielles et leurs valeurs.
            table_name (str): Nom de la table pour l'affichage (optionnel).
        """
        # Retrieve original column names
        app_data = st.session_state["train_data"]  # or pass Data object as a parameter
        feature_df = pd.DataFrame(X_train, columns=app_data.feature_columns)
        # Add category columns (if they aren't already present)
        for category_col, category_values in categories.items():
            if category_col not in feature_df.columns:
                feature_df[category_col] = category_values

        target_df = pd.DataFrame(Y_train, columns=app_data.target_columns)
        combined_df = pd.concat([feature_df, target_df], axis=1)

        st.write(f"### {table_name} Data Table")
        st.dataframe(combined_df)
        return combined_df

    @staticmethod
    def display_training_performance(model, encoder, data: Data):
        """
        Displays the training performance metrics.

        Args:
            model (RandomForestRegressor): Trained regression model.
            encoder (CategoryEncoder): Fitted CategoryEncoder.
            data (Data): Training data object.
        """
        # Transformation des données d'entrée avec l'encodeur
        X_encoded = encode_data(data, encoder)

        # Vérification des dimensions
        if X_encoded.shape[0] != data.Y.shape[0]:
            raise ValueError("Le nombre de lignes dans X_train et Y_train ne correspond pas.")

        # Prédictions sur l'ensemble d'entraînement
        Y_train_pred = model.predict(X_encoded)

        # Calcul de l'erreur quadratique moyenne (MSE) par cible
        mse_train = mean_squared_error(data.Y, Y_train_pred, multioutput='raw_values')

        # Construction d'un DataFrame pour un affichage clair
        mse_df = pd.DataFrame(
            {"MSE": mse_train},
            index=[f"Target_{i+1}" for i in range(data.Y.shape[1])]
        )

        # Affichage dans Streamlit
        st.write("### Performance on training data")
        st.table(mse_df)
        st.write("MSE mean:", mse_train.mean())

    @staticmethod
    def handle_predictions(model, encoder, data: Data) -> Optional[pd.DataFrame]:
        """
        Handles predictions on new data.

        Args:
            model (RandomForestRegressor): Trained regression model.
            data (Data): New data object.
        """
        if st.button("Predict targets"):
            predictions = predict_targets(model, encoder, data)
            st.write("### Prediction Results")
            predictions_df = pd.DataFrame(predictions, columns=data.target_columns)
            st.dataframe(predictions_df, width=1000)
            return predictions_df
        else:
            return None
