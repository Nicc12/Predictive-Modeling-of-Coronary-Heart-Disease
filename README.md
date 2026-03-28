# [Heart Disease Calculator Application 💓]: Predicting patient heart disease risk with Machine Learning

### 📋 Overview

The Goal: Built a fully-functioning heart disease calculator that predicts a patient's likelihood of having heart disease
The Impact: 88% Accurate heart disease calculator application that can assist with diagnosis. 

### Tech & Methods ⚙️:

![Static Badge](https://img.shields.io/badge/Python%20-%20blue?style=for-the-badge&logo=Python&color=gold) ![Static Badge](https://img.shields.io/badge/Machine%20Learning%20-%20blue?style=for-the-badge)
* Languages/Tools: Python, Visual Studio Code, Streamlit
* Libraries: Streamlit, ucimlrepo, pandas, seaborn, matplotlib, sklearn, statsmodels, numpy
* Concepts: Multiple Logistic Regression, Random Forest, Decision Tree, Data Pipelines
  
### 🗺️ Project Architecture

* Data Ingestion: The data was collected by the University of California, Irvine, and stored in a UCI repository. The data was imported directly into a Jupyter Notebook using the ucimlrepo Python package. 
* Processing/Storage: The data was cleaned in Python. 
* Analysis/Modeling: Multiple Logistic Regression and Random Forest machine learning methods were implemented to predict likihood of Coronary Heart Disease.
* Delivery: Streamlit was used to develop an interactive predictor application that takes in customer data to then output likinhood of heart disease and a comparative dashboard.

### 📊 Key Insights & Outcomes
Insight 1: Identified Major Vessels, Chest Pain, and Sex as key predictors of heart diseases.
Insight 2: Caveats with omitted variable bias exist. Supported by Gary D. Friedman in a 1975 study, a correlation exists between Cigarette Smoking and Chest Pain, leading to variables that are accounted for. Further data collection should identify habitual information to achieve higher accuracy.

### 📂 Repository Structure

* Data: [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease)
* Notebooks: Predictive Modeling of Coronary Heart Disease.ipynb
* src: app.py
* final products: [APP](https://share.streamlit.io/user/nicc12)
