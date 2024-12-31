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
        st.session_state["page"] = "Home"

    # Create pages
    st.sidebar.title("Multi-Output Regression Application with Clusters")
    home_button = st.sidebar.button("🏠 Load Data", key="home_button", use_container_width=True)
    analyses_button = st.sidebar.button("📊 Calcul & Show Results", key="analyses_button", use_container_width=True)

    if home_button:
        st.session_state["page"] = "Home"
    elif analyses_button:
        st.session_state["page"] = "Analyses"

    page = st.session_state["page"]

    if page == "Home":
        st.title("Home")
        train_file = st.file_uploader("Upload `training` data file", type=["csv"])
        test_file = st.file_uploader("Upload `test` data file", type=["csv"])

        if train_file and test_file:
            train_data = Data.load_from_file(train_file)
            test_data = Data.load_from_file(test_file)
            st.session_state["training_data_generated"] = True
            st.session_state["train_data"] = train_data
            st.session_state["test_data"] = test_data

        # Select clustering column
        if st.session_state["training_data_generated"]:
            categorical_columns = st.session_state["train_data"].category_columns
            clustering_column = st.selectbox("Select clustering column:", categorical_columns)
            st.session_state["clustering_column"] = clustering_column

    elif page == "Analyses":
        st.title("Analyses")
        # Run the application with training data
        if st.session_state["training_data_generated"]:
            app = App(config, st.session_state["train_data"], st.session_state["test_data"])
            app.run()
        else:
            st.write("Please upload the training and test data on the Home page.")
