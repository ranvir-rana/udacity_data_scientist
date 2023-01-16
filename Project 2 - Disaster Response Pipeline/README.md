# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installation](#installation)
	3. [Execution](#execution)
	4. [Additional Material](#material)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Description

This project is part of the Data Science Nanodegree program by Udacity in collaboration with Figure Eight. The goal is to build a Natural Language Processing (NLP) model to categorize messages from real-life disaster events in real-time. The dataset contains pre-labelled tweets and messages.

The project is divided into the following key sections:

    1. Processing data, building an ETL pipeline to extract, clean and store data in a SQLite DB.
    2. Building a machine learning pipeline to train a model that can classify text messages in various categories.
    3. Running a web app that displays model results in real-time.

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies

* Python 3.5+
* NumPy, SciPy, Pandas, Scikit-Learn for Machine Learning
* NLTK for Natural Language Processing
* SQLalchemy for SQLlite Database
* Pickle for Model Loading and Saving
* Flask, Plotly for Web App and Data Visualization

<a name="installation"></a>
### Installation

To clone the git repository:
```
git clone https://github.com/ranvir-rana/udacity_data_scientist.git
```

<a name="execution"></a>
### Execution

1. Run the following commands in the project's directory to set up the database, train model and save the model.

    - To run ETL pipeline for data cleaning and storing data in the database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
2. Run the following command in the app's directory to run the web app:
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="material"></a>
### Additional Material

In the data and models folder, you can find two Jupyter notebooks that will help you understand how the model works step by step:

    * ETL Preparation Notebook: learn about the ETL pipeline implemented
    * ML Pipeline Preparation Notebook: examine the Machine Learning Pipeline developed with NLTK and Scikit-Learn.

<a name="authors"></a>
## Authors

* [Ranvir Rana](https://github.com/ranvir-rana)

<a name="license"></a>
## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model
