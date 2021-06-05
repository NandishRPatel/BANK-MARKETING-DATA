## EXECUTIVE SUMMARY 

Finance and banking are one of the most extensive and extremely competitive markets. Any organisation going face-to-face with the big players need to fortify that they understand precisely how buyers like to interact with their sales and marketing processes. Customers today lean on both online as well as offline means to 
shop before making a decision. Almost on every occasion, an offline phenomenon such as making a phone 
call or visiting a branch is a positive indicator of a possible conversion. In this report, a bank marketing dataset 
of a Portuguese bank is selected where the marketing campaigns were based on phone calls. The report is 
based on the sample of the entire dataset which is about 50% of the original dataset. Initially, the dataset is 
split into training data, testing data and evaluation data. The outcome of the campaign is predicted using 
various classification and prediction models. The models used for predicting the result are Decision trees 
(C5.0, CART, QUEST, CHAID, RANDOM TREES), Neural Networks, Bayesian network and Support 
Vector Machines (SVM). Each model is presented briefly in this report and the best model is selected for 
analysis. C5.0 produced the best results among all models. 


## INTRODUCTION 
Banks and financial institutions exist to offer financial services to people and to make huge profits. Having 
said that, banks also devote remarkable resources and business intellect to gain capital. One of the most 
common ways for banks to do this is to engage in direct marketing campaigns like phone calls and face-to-face meetings to 
promote and provide services. Phone calls, i.e. Telemarketing is a conventional marketing  technique that helps to soar profits 
for any given business. Moreover, it also offers a more interactive and personal medium of sale service which can initiate 
an instant rapport with the prospective customers. Furthermore, telemarketing can help an organization to reach out more 
customers than with in-person or by going door-to-door and it can benefit a company to sell a product to both existing and 
new customers. For banking industry, telemarketing can be useful to communicate with large number of customers and offer them 
with all the services that they have for them. This may include, information about loans, term deposits, 
mortgages, Overdraft facility, Credit cards etc. 

For this project, a data set of a Portuguese Bank direct marketing campaign is used. This dataset is obtained 
from the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The 
primary objective of this project is to find the best model to predict whether a customer will subscribe for a 
term deposit or not using various classification techniques. Our secondary objective is to determine what 
factors in this data set would contribute the most for the sale of term deposits to the potential customers. The 
target users for this project are the marketing team of a banking institution who are looking to increase their 
inflow of cash deposits. The following sections of this report includes the description of the dataset in detail, 
all the methods (classification techniques) that has been applied on the dataset to get the results and eventually 
the best one is described thoroughly. Moreover, the results for the best model is presented followed by 
conclusion which summarises the most important findings and the scope of future research is suggested. 


## CONCLUSION

This project is focused on data mining techniques through which an ideal model is produced with the goal of 
enhancing marketing approach for a bank. 
The key findings from this project are: 
* C5.0 is the most suitable model for this data. The best predictive factor to attract a client is the duration 
of the call. More the duration means it is very likely that the client will subscribe a term deposit. 
* The outcome of the previous campaign creates a massive impact on the result. If the previous outcome 
is success, a client is 90% likely to opt for the term deposit again. 
* The ideal months for contacting a client is during the mid-year (May to August). During these months, 
71% of the people have subscribed for term deposit. 
We believe that the bank campaign management can attract a potential client by initiating a friendly rapport 
with them and convince them to opt for term deposits by engaging them in conversation. Furthermore, the 
bank needs to focus during the mid-year by offering new schemes to the client. For future research, we suggest 
that latest data should be used to yield a better model, ideally, by adding several characteristics which are not 
associated to the contact execution such as demographics of the client (eg: place of residence, annual income 
etc). 
