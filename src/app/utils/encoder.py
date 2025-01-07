import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class CategoryEncoder:
    def __init__(self, encoding_type="label"):
        """
        Initialiser l'encodeur.
        
        Args:
        - encoding_type (str): "label" pour Label Encoding ou "onehot" pour One-Hot Encoding.
        """
        if encoding_type not in ["label", "onehot"]:
            raise ValueError("encoding_type must be 'label' or 'onehot'")
        self.encoding_type = encoding_type
        self.encoders = {}  # Dictionnaire pour stocker les encodeurs par colonne

    def fit_transform(self, df, categorical_columns):
        """
        Encoder les colonnes catégorielles spécifiées.
        
        Args:
        - df (pd.DataFrame): Le DataFrame contenant les données.
        - categorical_columns (list): Liste des noms des colonnes catégorielles à encoder.
        
        Returns:
        - pd.DataFrame: DataFrame avec les colonnes encodées.
        """
        df_encoded = df.copy()  # Copie du DataFrame original
        for col in categorical_columns:
            if self.encoding_type == "label":
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.encoders[col] = le  # Stocker le LabelEncoder pour une éventuelle utilisation ultérieure
            elif self.encoding_type == "onehot":
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_array = ohe.fit_transform(df_encoded[[col]])
                encoded_df = pd.DataFrame(
                    encoded_array, 
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                    index=df_encoded.index
                )
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
                self.encoders[col] = ohe  # Stocker le OneHotEncoder
        return df_encoded

    def transform(self, df):
        """
        Appliquer les encodeurs entraînés sur de nouvelles données.
        
        Args:
        - df (pd.DataFrame): Le DataFrame à transformer.
        
        Returns:
        - pd.DataFrame: DataFrame avec les colonnes encodées.
        """
        df_encoded = df.copy()  # Copie du DataFrame original
        for col, encoder in self.encoders.items():
            if isinstance(encoder, LabelEncoder):
                df_encoded[col] = encoder.transform(df_encoded[col])
            elif isinstance(encoder, OneHotEncoder):
                encoded_array = encoder.transform(df_encoded[[col]])
                encoded_df = pd.DataFrame(
                    encoded_array, 
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                    index=df_encoded.index
                )
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
        return df_encoded
