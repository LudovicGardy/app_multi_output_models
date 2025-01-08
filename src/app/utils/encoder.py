import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class CategoryEncoder:
    def __init__(self, encoding_type="label"):
        """
        Initialize the encoder.
        
        Args:
        - encoding_type (str): "label" for Label Encoding or "onehot" for One-Hot Encoding.
        """
        if encoding_type not in ["label", "onehot"]:
            raise ValueError("encoding_type must be 'label' or 'onehot'")
        self.encoding_type = encoding_type
        self.encoders = {}  # Dictionary to store encoders by column

    def fit_transform(self, df, categorical_columns):
        """
        Encode the specified categorical columns.
        
        Args:
        - df (pd.DataFrame): The DataFrame containing the data.
        - categorical_columns (list): List of names of the categorical columns to encode.
        
        Returns:
        - pd.DataFrame: DataFrame with encoded columns.
        """
        df_encoded = df.copy()  # Copy of the original DataFrame
        for col in categorical_columns:
            if self.encoding_type == "label":
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.encoders[col] = le  # Store the LabelEncoder for potential future use
            elif self.encoding_type == "onehot":
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_array = ohe.fit_transform(df_encoded[[col]])
                encoded_df = pd.DataFrame(
                    encoded_array, 
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                    index=df_encoded.index
                )
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
                self.encoders[col] = ohe  # Store the OneHotEncoder
        return df_encoded

    def transform(self, df):
        """
        Apply the trained encoders to new data.
        
        Args:
        - df (pd.DataFrame): The DataFrame to transform.
        
        Returns:
        - pd.DataFrame: DataFrame with encoded columns.
        """
        df_encoded = df.copy()  # Copy of the original DataFrame
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
