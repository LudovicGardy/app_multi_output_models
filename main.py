import streamlit as st
import os

from src.app.app import App
from src.app.utils.config import load_config
from src.app.utils.data import Data


if __name__ == "__main__":
    # Set the environment
    env = 'development'

    # Load the configuration
    config = load_config(f'{env}/settings.yaml')

    # Initialize session state if not already initialized
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False
        st.session_state["train_data"] = None
        st.session_state["test_data"] = None
        st.session_state["page"] = "Home"

    # Create pages
    st.sidebar.title("Multi-Output Regression")
    home_button = st.sidebar.button("🏠 Home", key="home_button", use_container_width=True)
    analyses_button = st.sidebar.button("📊 Results", key="analyses_button", use_container_width=True)

    if home_button:
        st.session_state["page"] = "Home"
    elif analyses_button:
        st.session_state["page"] = "Analyses"

    page = st.session_state["page"]

    if page == "Home":
        st.title("🏠 Home")
        st.write("### Please upload the training and test data files or use default")
        st.write("  - The application will automatically detect the categorical columns.")
        st.write("  - You must name the target columns with the word 'target'.")
        st.write("  - See the example data files for more information.")
        
        data_option = st.radio("Choose data loading option:", ("Upload manually", "Use default data"))

        if data_option == "Upload manually":
            with st.container(border=True):
                st.write("### Load data")
                train_file = st.file_uploader("Upload **training** data file", type=["csv"])
                test_file = st.file_uploader("Upload **test** data file", type=["csv"])

            if train_file and test_file:
                train_data = Data.load_from_file(train_file)
                test_data = Data.load_from_file(test_file)
                st.session_state["data_loaded"] = True
                st.session_state["train_data"] = train_data
                st.session_state["test_data"] = test_data

        elif data_option == "Use default data":
            train_file_path = os.path.join("data", "generated_data_train.csv")
            test_file_path = os.path.join("data", "generated_data_test.csv")
            if os.path.exists(train_file_path) and os.path.exists(test_file_path):
                train_data = Data.load_from_file(train_file_path)
                test_data = Data.load_from_file(test_file_path)
                st.session_state["data_loaded"] = True
                st.session_state["train_data"] = train_data
                st.session_state["test_data"] = test_data
            else:
                st.error("Default data files not found in the 'data/' directory.")

        # Select clustering column
        if st.session_state["data_loaded"]:
            if st.session_state["train_data"].category_columns:
                categorical_columns = st.session_state["train_data"].category_columns
                clustering_column = st.selectbox("Select clustering column:", categorical_columns)
                st.session_state["clustering_column"] = clustering_column
            else:
                st.warning("No categorical columns found in the training data")
            if data_option == "Use default data":
                st.sidebar.info("Default data loaded successfully")
            else:
                st.sidebar.info("Data loaded successfully")
        else:
            st.sidebar.warning("No data loaded")

    elif page == "Analyses":
        st.title("📊 Analyses")
        # Run the application with training data
        if st.session_state["data_loaded"]:
            app = App(config, st.session_state["train_data"], st.session_state["test_data"])
            app.run()
        else:
            st.error("Please upload the training and test data on the Home page, or select the default dataset.")
