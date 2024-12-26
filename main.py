import streamlit as st

from src.app.app import App
from src.app.utils.config import load_config
from src.app.utils.data import Data


if __name__ == "__main__":
    # Set the environment
    env = 'development'

    # Load the configuration
    config = load_config(f'{env}/settings.yaml')

    # Initialize session state if not already initialized
    if "training_data_generated" not in st.session_state:
        st.session_state["training_data_generated"] = False
        st.session_state["train_data"] = None
        st.session_state["test_data"] = None

    # Generate data
    if not st.session_state["training_data_generated"]:
        train_data = Data.generate(n_samples=200)
        test_data = Data.generate(n_samples=20)

        st.session_state["training_data_generated"] = True
        st.session_state["train_data"] = train_data
        st.session_state["test_data"] = test_data
    else:
        train_data = st.session_state["train_data"]
        test_data = st.session_state["test_data"]

    # Run the application with training data
    app = App(config, train_data, test_data)
    app.run()
