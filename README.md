# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Executing Program](#executing)
	4. [Additional Material](#material)
3. [Authors](#authors)
4. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Description

The goal for this project is to build a webapp that classifies disaster messages into categories, enabling authorities to send messages to appropriate disaster relief agencies. This project includes building an ETL Pipeline, a Machine Learning Pipeline, and a web app to display results and visualize data.

This project is in partial fulfillment of the Udacity Data Science Nanodegree Course requirements. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+
* Data Wrangling: NumPy, Pandas, re 
* Machine Learning: Scikit-Learn
* Natural Language Process libraries: nltk
* SQLite Database libraries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly 

A more detailed requirements.txt file is uploaded.

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
git clone https://github.com/noemistatcat/Disaster_Response_Pipeline
```
<a name="executing"></a>

### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Additional Material

Jupyter notebooks that detail step-by-step process of running the models are also available:
1. **ETL Pipeline Preparation**: An ETL Pipeline that takes in the raw message and raw categories, and outputs a merged, cleaned and transformed database. 
2. **ML Pipeline Preparation**: Takes in the cleaned database, performs preprocessing on text data, and then loads it into a machine learning pipeline to create a classification model.

<a name="authors"></a>
## Authors

* [Noemi Ramiro](https://github.com/noemistatcat)

## Acknowledgements

I thank [Udacity](https://www.udacity.com/) for providing this challenge and learning experience. I also acknowledge [Figure Eight](https://www.figure-eight.com/) for providing the messages dataset to train my model.
