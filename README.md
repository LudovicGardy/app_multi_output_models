# Multi Output Regression

## 📄 Overview
🏡 Para 1  
This application is a comprehensive multi-output analysis tool that allows you to load datasets, handle multiple target columns, and quickly visualize results.

🤔 Para 2  
It integrates seamlessly into a machine learning workflow by offering an interactive Streamlit interface, suitable for data science or data engineering teams looking to test different scenarios.

Para 3  
You can customize encodings, manipulate various datasets, and extend the application to more advanced regression methods as needed.

🌐 Access the app and start your exploration now at [@Not implemented yet](https://wikipedia.com).

![Image1](images/image1.png)

## Prerequisites

- **Anaconda or Miniconda**: For easy environment management (optional)
- **Docker** (optional): To run the app in a containerized setup
- **Python 3.12**

## ⚙️ Installation and Setup

This project uses **Poetry** for dependency management, which offers an option to create virtual environments directly. You have two choices to set up your environment:

### Option 1: Using Poetry to Create and Manage the Virtual Environment

Poetry can handle the creation and activation of a virtual environment automatically, without needing Conda or other virtual environment tools.

#### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LudovicGardy/the_project_name
    cd the_project_name
    ```

2. **Install Dependencies and Create the Environment**:
   - If there is no `pyproject.toml` file in your project yet, you can create one using:
     ```bash
     poetry init
     ```

   - Add the libraries you need, for exemple:
     ```bash
     poetry add matplotlib
     ```

   - Run the following command to let `poetry` create an isolated environment and install dependencies directly:
     ```bash
     poetry install --no-root
     ```

3. **Activate the Environment**:
   - Poetry automatically creates a virtual environment, but it may not activate it by default. Use the following command to activate it:
     ```bash
     poetry shell
     ```

4. **Run the Application**:
   1. Classic application
      ```bash
      python main.py
      ```
   2. Streamlit application
      ```bash
      streamlit run main.py
      ```

#### Note on Poetry-managed Environments

Using `poetry install` to create and activate an environment directly is a streamlined approach, making Conda unnecessary. This can be ideal if you want a simple, Python-only virtual environment managed by `poetry`.

---

### Option 2: Using Conda for Virtual Environment Management

If you prefer using Conda, follow these steps:

#### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LudovicGardy/the_project_name
    cd the_project_name
    ```

2. **Create and Activate the Conda Environment**:
   - **Using Conda** (recommended):
       ```bash
       conda create -n myenv python=3.11
       conda activate myenv
       ```

3. **Install Dependencies**:
   1. **With Poetry**
   - Once the environment is active, install dependencies within it:
     ```bash
     poetry install --no-root
     ```
   1. **Or with pip**
      ```bash
      pip install -r requirements.txt
      ```

4. **Run the Application**:
   1. Classic application
      ```bash
      python main.py
      ```
   2. Streamlit application
      ```bash
      streamlit run main.py
      ```

---

### Option 3: Run the App with Docker

If you prefer to run the application in a containerized setup, use Docker.

#### Steps

1. **Set Up Docker Environment**

   - Make sure **Docker** is installed and running on your system.

2. **Navigate to the Correct Directory**

   - **For a multi-container setup**:
     ```bash
     cd [path-to-app-folder-with-docker-compose.yml]
     ```

   - **For a single container**:
     ```bash
     cd [path-to-app-folder-with-Dockerfile]
     ```

3. **Build and Start the Containers**

   - Run the following command to build and launch the app in Docker:
     ```bash
     docker-compose up --build
     ```

   - **Access the Application**:
     - Open your browser and go to `http://localhost:8501`.

   - **Troubleshooting**:
     - If there’s an issue with `pymssql`, try adjusting its version in `requirements.txt` or temporarily removing it.

---

## 🧪 Running Unit Tests

Poetry can also manage unit tests to help you verify that the application functions as expected. This project uses **pytest** for testing.

### Setting Up and Running Tests

1. **Add Testing Dependencies**:

   If `pytest` and other test dependencies are not yet added, include them using Poetry:
   ```bash
   poetry add --dev pytest
   ```

2. **Writing Tests**:

   - Place your test files in a `tests` directory at the root of the project.
   - Name your test files following the pattern `test_*.py`.

3. **Running Tests**:

   To run tests within the Poetry environment, use:
   ```bash
   poetry run pytest
   ```

   Alternatively, if you are in an activated Poetry shell (`poetry shell`), you can simply run:
   ```bash
   pytest
   ```

4. **Viewing Test Results**:

   - Test results will appear in the terminal, with a summary of passed, failed, and skipped tests.
   - Use `pytest -v` for more detailed output.

---

## 👤 Author

- **LinkedIn**: [Ludovic Gardy](https://www.linkedin.com/in/ludovic-gardy/)
- **Website**: [https://www.sotisanalytics.com](https://www.sotisanalytics.com)