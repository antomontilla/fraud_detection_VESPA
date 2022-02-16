# fraud_detection_VESPA
Project for fraud detection in card transactions

# Fraud detection in card transactions -- Project Overview
* This project develops models for predicting whether a transaction is fraudulent or not, through the analysis and manipulation of a database that contains information on 590,540 transactions carried out online through the Vesta platform.
* The project evaluates the most relevant variables to predict the veracity of a transaction, builds models and selects the one that best fits the data.
* As a general conclusion, it was identified that the classification algorithms of XGBClassifier, GradientBoostingClassifier and Random Forest are the ones that best fit the data, giving rise to a deeper development of this type of models in future works.
* For its further application, the distribution of the target variable must be taken into account, and ensure that the dataframe to be used in the modelling is balanced.
* Once the modelling process has been improved, the next task would be the development of algorithms that, based on the prediction of the veracity of a transaction, instruct mechanical tasks for its validation by the user. An example would be developing an application that automatically checks with the user if they are carrying out the transaction in question.

## Data gathering process
This project is based on the analysis and manipulation of a database provided by Vesta Corporation, available in Kaggle (see [here](https://www.kaggle.com/c/ieee-fraud-detection)). Vesta corporation provides a platform for conducting financial operations online. In this challenge, the company instructs participants to propose models for detecting fraudulent online transactions.

The database includes information on 590,540 transactions carried out online through the Vesta platform. Specifically, the database includes 434 columns for each observation, detailing information regarding the transaction itself (such as product code, date, payment method) and the identity of the buyer (such as consumption habits, location, device used, among others). Due to data security restrictions, the columns referring to identity are hidden.

The target variable is IsFraud, which takes binary values (0, 1).

## Initial data exploration
Due to technological infrastructure limitations, the project has been developed entirely on the Kaggle platform, which allows working in the cloud and amplifying resources in terms of memory and computational capacity versus the capacity of regular personal computers.

The first step of the project was based on an exhaustive exploration of the database. As a whole, the database contains 590,540 observations and 434 columns.

### Target variable
* The target variable takes values 0 (no fraud) and 1 (fraud).
* Only 2% of observations are fraud.
* In principle, the low proportion of observations with fraud relative to non-fraud could affect the robustness of the estimators in the modeling, in the sense that estimation biases could occur.

<img width="415" alt="image" src="https://user-images.githubusercontent.com/56187009/154331211-5a22a555-4ee7-420b-b9ef-7b8da0c7894f.png">

### Explanatory variables
The variables are grouped according to the following criteria:
* TransactionID, which represents the identifier of each transaction.
* TransactionDT, indicates time since transaction (in seconds).
* TransactionAmt, indicates the amount of the transaction (in USD).
* ProductCD, the product code of the transaction. Categorical variable.
* Card (01-06), information about the payment card, such as type, category, bank, country, among others. Categorical variables.
* Addr (01-02), customer and vendor address. Categorical variables.
* Dist (01-02), refers to the distance between the location of the client and the seller.
* Pemaildomain, the buyer's email domain. Categorical variable.
* Remaildomain, the seller's email domain. Categorical variable.
* C (01-14), hidden columns with encrypted data. They refer to counting variables, such as the number of addresses associated with the card, telephone numbers, email addresses, among others, for both the buyer and the seller.
* D (01-15), time variables, such as the time since the last transaction, among others.
* M (01-09), variables that indicate if there is a match between the purchase information. Categorical variables.
* V (01-339), variables provided by Vesta regarding the transaction, such as classification, count, among others.
* Id (01-138), hidden variables with encrypted data. They refer to identity data, which for data protection reasons cannot be disclosed. Personal data of the seller and buyer, data of the connection or equipment (IP, ISP, proxy).
* Devicetype, type of device used by the buyer. Categorical variable.
* Deviceinfo, information of the device used by the buyer. Categorical variable.

