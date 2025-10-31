# Jordan Project 02
### Author:  [JfromNWMS](https://github.com/JfromNWMS)

## Overview
This project uses seaborn's Titanic dataset to demonstrate pre-processing of data before splitting for analysis and to compare splitting methods.  The dataset is cleaned and inspected before comparisons between standard pseudo-random splitting and stratified spliting are made.

This project is a guided notebook assignment that demonstrates how to:
- Load and explore a dataset.
- Choose and justify features for predicting a target variable.
- Check for class distributions of pseudo-random and stratified test-train-split methods.
- Document work in a structured Jupyter Notebook.

A link to the notebook can be found here: [ml02_jordan.ipynb](https://github.com/JfromNWMS/applied-ml-jordan/blob/main/notebooks/project02/ml02_jordan.ipynb)

This project utilizes a module for calculating Tukey's fences.  The module can be found here: [stats_jordan.py](https://github.com/JfromNWMS/applied-ml-jordan/blob/main/src/stats_jordan.py)

## Dataset  
- We use the built-in dataset from seaborn:  
   - [Titanic-Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)  

## Python Library for Machine Learning: scikit-learn
We use scikit-learn, built on NumPy, SciPy, and matplotlib
   - Read more at <https://scikit-learn.org/>
   - Scikit-learn supports classification, regression, and clustering.
   - This project applies regression.


## Professional Python Setup and Workflow
We follow professional Python practices. 
Full instructions are available at <https://github.com/denisecase/pro-analytics-02/>. 


**Important:** VS Code + Pylance may fail to recognize installed packages in Jupyter notebooks.  
See the above guides for troubleshooting and solutions.  

---

## Project Outline
This Machine learning project follows a structured approach.

### Section 1. Load and Explore the Data
- 1.1 Load the Dataset
- 1.2 Preliminary Inspection
- Reflection 1:
How many data instances are there?
How many features are there?
What are the names?
Are there any missing values?
Are there any non-numeric features?
Are the data instances sorted on any of the attributes?
What are the units of age?
What are the minimum, median and max age?
What two different features have the highest correlation?
Are there any categorical features that might be useful for prediction?

### Section 2. Data Exploration and Preparation
- 2.1 Create Scatter Matrix, Scatter Plot, Histogram and Count Plot
- 2.2 Identify patterns or anomalies in feature distributions.
- 2.3 Feature Engineering
- Reflection 2.3
Why might family size be a useful feature for predicting survival?
Why convert categorical data to numeric?

### Section 3. Choose Features and Target
- Reflection 3:
Why are these features selected?
Are there any features that are likely to be highly predictive of survival?

### Section 4. Splitting
- Reflection 4:
Why might stratification improve model performance?
How close are the training and test distributions to the original dataset?
Which split method produced better class balance?

See [EXAMPLE_ANALYSIS](./EXAMPLE_ANALYSIS.md) for more.

---

