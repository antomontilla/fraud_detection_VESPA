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

## Simplification of the project and data exploration
The exhaustive exploration of the explanatory variables by group allowed: i) to study their behaviour, ii) the relationship with the objective variable, and iii) to make preliminary decisions on whether to maintain/transform them for modelling.

The first step was the verification of null and empty observations for possible deletion. In total, 232 columns contained more than 40% of empty observations; the remaining 202 columns showed more than 70% of the loaded data, i.e. as non-null. 

As a first step in the reduction of the database, the 232 columns with more than 40% null values were eliminated. As a next step, the null observations were eliminated. The cleaning process allowed the database to be reduced to 202 columns and 328,198 observations.

For the Exploratory Data Analysis (EDA) process, due to the size of the database, it was decided to extract a sample of 15% (50,000 data) of the total to facilitate data exploration.

Among the most relevant points in this section:
* Fraudulent transactions, on average, tend to be for higher amounts (around 50%) than non-fraudulent transactions, as would be expected a priori.

<img width="415" alt="image" src="https://user-images.githubusercontent.com/56187009/154332109-a50dbd72-30db-497b-ba0d-bb0aec105f8a.png">

* It is decided to eliminate the following explanatory variables (due to the lack of variability, both in terms of data distribution and in its relationship with IsFraud): ProductCD, addr2, P_emaildomain, Card3.
* In terms of the variables containing the features of the card, it is observed that fraudulent transactions tend to be carried out more frequently by credit cards, although there is not much distinction between whether it is with a visa or a mastercard.

<img width="403" alt="image" src="https://user-images.githubusercontent.com/56187009/154332189-d54579d9-e0f8-44f8-a5a7-49be8b5ae9ec.png">

* The address of both the buyer and the seller, as well as the domain of the email addresses used, do not seem to add much in terms of whether a transaction is fraudulent.
* For the set of columns with encrypted data and without title or content information, including C 01-14 (count variables) and V 01-339 (unknown transaction variables), it was decided to use a (PCA) principal components analysis. This methodology allowed selecting a limited number of components (linear combinations of each set of variables) and eliminating the original columns.
* The se of variables denoting time tend to exhibit significant differentiation from the target variable IsFraud.

On balance, the exploratory analysis allowed to reduce the database from 202 columns to only 39, including categorical columns which were transformed into factors and PCA components, which will help in the modelling and evaluation process.

The risk that the database is unbalanced was also identified: the target variable includes only 2% of the observations as fraud. This could cause certain problems in the modelling process.

## Correlation matrix

<img width="415" alt="image" src="https://user-images.githubusercontent.com/56187009/154332305-7e9be20b-e11d-4a21-ac84-9c845e5e1c50.png">

* The initial analysis suggests that the target variable correlates more with the set of columns D, type of card (debit or credit), and some principal components of columns C and V.
* Similarly, it can be seen that there are certain columns that do not seem to be related to the target variable and it could be useful to extract them from the database.
* When studying the correlation matrix only for the top 15 of the most correlated columns, it is observed that from column 13, the correlation coefficient with IsFraud falls below 3%.
* It was decided to take these first 13 columns for the modelling process.

## Methodology and modelling
Due to the amount of data and the (binary) distribution of the target variable, classification and clustering models are selected. Specifically, the following models were applied:

* Logistic regression.
* Decision tree.
* Random forest.
* K-nearest neighbors.
* Gaussian Naïve Bayes.
* GradientBoostingClassifier.
* XGBClassifier model.

### Modelling 1.0
In this first phase, the selected models are applied to the database obtained in the EDA, which includes 13 explanatory columns and 328,198 observations, to estimate IsFraud. 

For computational reasons, a sample of 100,000 observations is drawn. The database is divided into train and test with a criterion of 80-20 and with the seed set at 42 for the validation of the models.

**ROC Curve Models 1.0**
<img width="415" alt="image" src="https://user-images.githubusercontent.com/56187009/154332692-7765e416-4068-4093-87f0-564ba6b3dc5a.png">

The Gaussian Naive Bayes classification model is the one that best explains the variability of the data, with an area on the ROC curve of 63%, followed by the decision tree.

**ROC Curve Gaussian Naive Bayes model**
<img width="357" alt="image" src="https://user-images.githubusercontent.com/56187009/154332855-be05bfb2-2e5c-4a4c-9c30-3e5d72da2813.png">

However, the results reveal a bias in the estimators towards non-fraudulent transactions, which concentrate almost all of the successes of the model.

**Diffusion matrix Gaussian Naïve Bayes model**
<img width="189" alt="image" src="https://user-images.githubusercontent.com/56187009/154333014-10811e00-b667-4905-a52f-b7d1653cd61f.png">

* A possible reason could be the distribution of the target variable in the original dataframe: fraud represents 2% of total transactions.
* In fact, for the logistic regression and GradientBoostingClassifier models, the area under the ROC curve is 50%, that is, the models would randomly predict IsFraud.
* As indicated, this could correspond to the inadequacy of this type of algorithm to this type of data or other problems more typical of the dataframe used, such as the IsFraud imbalance.


### Modelling 2.0
In this section, I proceed to re-estimate the models but this time selecting a 'balanced' sample from the df_model dataframe.

The main problem with the original database is that the target variable is almost entirely distributed in non-fraudulent observations, which represent 98% of the total data. This flaw implies that the algorithms used tend to produce estimators with a bias towards transactions that are not frauds, as was observed in the diffusion matrix.

The undersampling procedure is based on: i) selecting the total number of observations where the target variable is equal to 1, ii) randomly selecting an equal number of observations where the target variable is equal to 0, iii) estimating the models, iv) compare and evaluate.

A caveat to take into account is the significant loss of data: the sample is reduced to approximately 13,000 observations, which could affect the quality of the estimated models.

**ROC Curve Models 2.0**
<img width="415" alt="image" src="https://user-images.githubusercontent.com/56187009/154333176-a329d820-8395-40b4-83f6-8c484ffd590c.png">

* In general, a significant improvement is observed in the predictive capacity of the selected algorithms in explaining the target variable, including in those algorithms that in the first section failed to produce statistically significant estimators.
* Likewise, the study of the confusion matrix reveals an almost equitable distribution in the correct answers of the model with respect to the objective variable: the correct answers of type (1,1) are generally almost equal to the correct answers (0,0).
* The results confirm the relevance of applying dataframe balancing methods.
* With 77% area in the ROC curve, the XGBClassifier model estimator has been the one that best fits the data, with a balanced distribution in terms of transaction type.

**ROC Curve XGBClassifier model**

<img width="306" alt="image" src="https://user-images.githubusercontent.com/56187009/154333288-eec28d25-dcca-4951-b190-c166ee1e5fd6.png">

**Diffusion matrix XGBClassifier model**

<img width="207" alt="image" src="https://user-images.githubusercontent.com/56187009/154333328-469433a0-360d-4866-92f7-2af2b0736272.png">

## Final observations and conclusions
* Being able to predict the veracity of a transaction made through a non-traditional method of payment is a fundamental objective to consolidate the advancement of technology in 21st century consumption.
* The information provided and stored by users, although abundant, complex and sensitive, is the elementary input to guarantee the proper functioning of fraud prediction models. Understanding, first, and treating, second, this information is the determining factor in the implementation of the machine learning/big data methodology in the prevention of bank fraud.
* In this project, it was identified that the classification algorithms of XGBClassifier, GradientBoostingClassifier and Random Forest are the ones that best fit the data, giving rise to a deeper development of this type of models in future works. For its further application, the distribution of the target variable must be taken into account, and ensure that the dataframe to be used in the modelling is balanced.
* An important deficit in this project has been the lack of knowledge of a large part of the variables with transaction information, in particular those referring to the identity and purchase/sale patterns of the buyer/seller, mainly due to regulatory restrictions.
* A next step, probably directly at the financial intermediaries, would be the appropriate exploitation of consumer information. Its extensive application in the real world would represent significant savings in the industry, both for the buyer and for the intermediary and the platform providers for the exchange of purchase/sale.
* Finally, once the modelling process has been improved, the next task would be the development of algorithms that, based on the prediction of the veracity of a transaction, instruct mechanical tasks for its validation by the user. An example would be developing an application that automatically checks with the user if they are carrying out the transaction in question.





