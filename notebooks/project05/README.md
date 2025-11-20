# Jordan Project 05
### Author:  [JfromNWMS](https://github.com/JfromNWMS)

## Overview
This project uses UCI Red Wine Quality dataset. We will preprocess the data and train Gradient Boosting and AdaBoost classifier ensemble models using features based on physicochemical tests to predict the wine quality rating given by wine tasters.  Then, comparisons of the models are given to each other and the models obtained from other team members.

This project is a guided notebook assignment that demonstrates how to:
- Load, explore and prepare a dataset.
- Choose and justify features for predicting a target variable.
- Predict a categorical target and evaluate performance of Gradient Boosting and AdaBoost ensemble models.
- Document work in a structured Jupyter Notebook.

A link to the notebook can be found here: [ensemble_jordan.ipynb](https://github.com/JfromNWMS/applied-ml-jordan/blob/main/notebooks/project05/ensemble_jordan.ipynb)

## Dataset  
- We use the Red Wine Quality dataset from UCI:  
   - [Red Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Model Summary

| Model                     |   Train Accuracy |   Test Accuracy |   Accuracy Gap |   Train F1 |   Test F1 |   F1 Gap |
|:--------------------------|-----------------:|----------------:|---------------:|-----------:|----------:|---------:|
| Gradient Boosting (Tuned) |           0.9875 |          0.8812 |         0.1062 |     0.9873 |    0.8631 |   0.1242 |
| AdaBoost (200, lr=0.1)    |           0.835  |          0.8594 |        -0.0243 |     0.7919 |    0.8262 |  -0.0344 |
| AdaBoost (200, lr=0.5)    |           0.8397 |          0.8562 |        -0.0165 |     0.816  |    0.833  |  -0.017  |
| Gradient Boosting (100)   |           0.9601 |          0.8562 |         0.1039 |     0.9584 |    0.8411 |   0.1173 |

## Instructions to Set up Virtual Environment

Run these commands in your VS Code terminal:

```shell
# 1. Create an isolated virtual environment
uv venv

# 2. Pin a specific Python version (3.12 recommended)
uv python pin 3.12

# 3. Install all dependencies, including optional dev/docs tools
uv sync --extra dev --extra docs --upgrade

# 4. Enable pre-commit checks so they run automatically on each commit
uv run pre-commit install

# 5. Verify the Python version (should show 3.12.x)
uv run python --version
```

Next, activate the virtual environment.

Windows (PowerShell):

```script
.\.venv\Scripts\activate
```

macOS / Linux / WSL:

```script
source .venv/bin/activate
```

## Instructions to Run Jupyter Notebook Locally

### Install the Jupyter Extension for VS Code:

Open the Extensions view in VS Code by pressing Ctrl+Shift+X (Windows/Linux) or Cmd+Shift+X (Mac).
Search for "Jupyter" and install the official extension.
Open the notebook in VS Code. The file will have a .ipynb extension.

### Task 1. Select Notebook Kernel
Open the notebook (.ipynb) file in VS Code.
If prompted, select a Python interpreter that corresponds to your project’s .venv.
If not prompted:
Click the Kernel Selector in the top-right corner.
Choose the interpreter labeled with your project name and path.
Or open the Command Palette (Ctrl Shift P / Cmd Shift P) and run: Python: Select Interpreter, then pick your .venv.

### Task 2. Start and Run the Notebook
To run notebooks directly in VS Code, click on a cell and press:
<br>Shift Enter to run cell and move to next
<br>Ctrl Enter to run cell and stay in place
<br>Save often or enable File / Auto Save.
