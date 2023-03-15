# Starbucks Capstone Challenge (Udacity - Data Science Nanodegree)

## Table of Contents
1. [Description](#description)
2. [Installations](#installations)
3. [Run](#run)
4. [Data](#data)
5. [Conclusion](#conclusion)
6. [Licensing, Authors, Acknowledgements](#acknowledgement)

<a name="descripton"></a>
## Description

The Udacity Data Science Nanodegree requires completion of a Capstone project which utilizes simulated data to mimic customer behavior on the Starbucks rewards mobile app. Starbucks sends out offers to users of the mobile app, which can be an advertisement or an actual offer such as a discount or BOGO. The data includes information on offers, customer demographics, and offer and transaction events.

The project goal is to use this data to classify whether an offer is likely to be successful based on demographic and offer information. This will be achieved by building a machine learning model that predicts offer success using demographic information and offer details from the data.

The motivation behind this project is to help Starbucks improve their mobile app's offer personalization to increase customer engagement and loyalty. By identifying which demographic groups respond best to which offer types, Starbucks can tailor their offers to each customer segment, thereby enhancing customer satisfaction and retention.

Additionally, the project can help Starbucks identify the most effective offer types and the demographic groups that are most likely to complete them. The findings can help Starbucks make informed decisions on how to allocate their marketing budgets and create targeted campaigns that resonate with each customer group.

The project will also answer the following questions:
-	Which offer is the most successful?
-	Do males or females spend more money?
	

### File Description
~~~~~~~
        Data Scientist Capstone - Starbucks Project
          |-- data
                |-- portfolio.json	# containing offer ids and meta data about each offe
                |-- profile.json	# demographic data for each customer
                |-- transcript.json		# records for transactions, offers received, offers viewed, and offers completed
          |-- Starbucks_Capstone_notebook.ipynb			# code file
          |-- README
~~~~~~~

<a name="installations"></a>
## Installations

This project requires **Python 3.x** and the following Python libraries installed:

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.


<a name="run"></a>
### Run

In a terminal or command window, navigate to the top-level project directory `Data Scientist Capstone - Starbucks Project/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Starbucks_Capstone_notebook.ipynb
```  
or
```bash
jupyter Starbucks_Capstone_notebook.ipynb
```

This will open the iPython Notebook software and project file in your browser.

<a name="data"></a>
### Data

The data used in this project is contained in three files: portfolio.json, profile.json, and transcript.json. These files can be found in the workspace.

1. portfolio.json

-	id (string) - offer id
-	offer_type (string) - type of offer, i.e., BOGO, discount, informational
-	difficulty (int) - minimum required spend to complete an offer
-	reward (int) - reward given for completing an offer
-	duration (int) - time for the offer to be open, in days
-	channels (list of strings) - channels used to send the offer

2. profile.json

-	age (int) - age of the customer
-	became_member_on (int) - date when the customer created an app account
-	gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
-	id (str) - customer id
-	income (float) - customer's income

3. transcript.json

-	event (str) - record description (i.e., transaction, offer received, offer viewed, etc.)
-	person (str) - customer id
-	time (int) - time in hours since start of test. The data begins at time t=0
-	value (dict of strings) - either an offer id or transaction amount depending on the record

<a name="conclusion"></a>
## Conclusion

In this project, we aimed to build a machine learning model that predicts offer success based on demographic and offer details provided in the data. We used several algorithms such as random forest, gradient boosting, AdaBoost, and logistic regression to build a simple classification model. The Gradient Boosting classifier had the highest accuracy of 71% for this task.

We identified membership duration, income, and age as the most relevant factors for offer success based on the model. Additionally, we answered the following questions:

* Which offer is the most successful?

The discount offer was found to be more successful than the BOGO offer based on a higher overall completed/received rate and a slightly higher absolute number of completed offers. However, the BOGO offer had a greater chance of being viewed or seen by customers.

* Who spends more money, male or female?

Female customers were found to spend more money than male customers based on the graph data.

Overall, the results of this project can help Starbucks optimize their marketing campaigns and improve offer success rates. There is a future scope to explore better modeling techniques and algorithms to improve the model's performance. An imputation strategy for missing values can also be used to see if the model can be improved in this way.

For more insights and a detailed analysis of the project, please refer to the blog <a href="https://medium.com/@ranvir_rana/forecasting-the-success-of-starbucks-offers-and-identifying-key-factors-for-success-43a1f4c7fc1a">link</a>.


<a name="acknowledgement"></a>
## Licensing, Authors, Acknowledgements 

* The data used in this project is provided by Starbucks.
* I give full credit to Starbucks for providing this data and allowing us to use it for this project.
* This project is part of the Data Scientist Nanodegree program offered by Udacity.
* I acknowledge Udacity for providing us with the necessary skills and knowledge to complete this project successfully.
* I thank our mentors and reviewers at Udacity for their guidance and feedback, which helped us improve our skills and complete the project successfully.
