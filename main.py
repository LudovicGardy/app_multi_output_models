import streamlit as st
import os

from src.app.app import App
from src.app.utils.config import load_config
from src.app.utils.data import Data


def load_data(train_file_path, test_file_path):
    train_data = Data()
    test_data = Data()
    train_data.load_from_file(train_file_path)
    test_data.load_from_file(test_file_path)
    st.session_state["data_loaded"] = True
    st.session_state["train_data"] = train_data
    st.session_state["test_data"] = test_data


def setup_session_state():
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False
        st.session_state["train_data"] = None
        st.session_state["test_data"] = None
        st.session_state["page"] = "Home"

    if "clustering_column" not in st.session_state:
        st.session_state["clustering_column"] = ""

    if "target_columns" not in st.session_state:
        st.session_state["target_columns"] = []


def handle_data_option(data_option):
    if data_option == "Upload manually":
        with st.container():
            st.write("### Load data")
            train_file_path = st.file_uploader("Upload **training** data file", type=["csv"])
            test_file_path = st.file_uploader("Upload **test** data file", type=["csv"])

        if train_file_path and test_file_path:
            load_data(train_file_path, test_file_path)

    elif data_option == "Use default data":
        train_file_path = os.path.join("data", "generated_data_train.csv")
        test_file_path = os.path.join("data", "generated_data_test.csv")
        if os.path.exists(train_file_path) and os.path.exists(test_file_path):
            load_data(train_file_path, test_file_path)
        else:
            st.error("Default data files not found in the 'data/' directory.")


def select_columns():
    available_columns = st.session_state["train_data"].all_columns

    clustering_column = st.selectbox("Select clustering column:", [""] + available_columns)
    st.session_state["clustering_column"] = clustering_column

    if not clustering_column:
        st.sidebar.warning("No clustering column selected")
    else:
        st.sidebar.success(f"Clustering column selected: {clustering_column}")

    target_columns = st.multiselect("Select target columns:", available_columns)
    st.session_state["target_columns"] = target_columns

    if not target_columns:
        st.sidebar.warning("No target columns selected")
    elif len(target_columns) < 2:
        st.sidebar.info(f"Amounts of targets selected: {len(target_columns)}")
    else:
        st.sidebar.success(f"Amounts of targets selected: {len(target_columns)}")

    feature_columns = [col for col in available_columns if col not in target_columns]
    st.session_state["feature_columns"] = feature_columns

    st.write("### Feature columns:")
    st.write(feature_columns)

    st.session_state["train_data"].encode_categorical_columns([clustering_column])
    st.session_state["test_data"].encode_categorical_columns([clustering_column])

    st.session_state["train_data"].get_feature_and_target_df(feature_columns, target_columns)
    st.session_state["test_data"].get_feature_and_target_df(feature_columns, target_columns)


if __name__ == "__main__":
    env = 'development'
    config = load_config(f'{env}/settings.yaml')

    setup_session_state()

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

        st.session_state["selected_model"] = st.radio("Choose a model:", ("Random Forest", "CatBoost"))
        data_option = st.radio("Choose data loading option:", ("Upload manually", "Use default data"))
        handle_data_option(data_option)

        if st.session_state["data_loaded"]:
            if data_option == "Use default data":
                st.sidebar.success("Default data loaded successfully")
            else:
                st.sidebar.success("Data loaded successfully")

            select_columns()
        else:
            st.sidebar.warning("No data loaded")

    elif page == "Analyses":
        st.title("📊 Analyses")
        if not st.session_state["data_loaded"]:
            st.error("Please upload the training and test data on the Home page, or select the default dataset.")
        if not st.session_state["clustering_column"]:
            st.error("No clustering column selected")
        if len(st.session_state["target_columns"]) < 2:
            st.error("No target columns selected. Select at least 2 targets.")
        else:
            app = App(config, st.session_state["train_data"], st.session_state["test_data"])
            app.run()