<H3>NAME : SIVAHARIBALAN K </H3>
<H3>REGISTER NO: 212224220103</H3>
<H3>EX NO:1</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```Python
import pandas as pd                                                
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         
df.head()
```
```Python
df.isnull().sum()                                                   
```
```Python                                               
df.duplicated().sum()                                             
```
```Python                                              
df=df.drop(['Surname', 'Geography','Gender'], axis=1)               
scaler=StandardScaler()                                             
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
```
```Python
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values                     
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)   
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)                    
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                
```


## OUTPUT:
### DATASET:
<img width="1593" height="387" alt="Screenshot (159)" src="https://github.com/user-attachments/assets/28c96cc0-6fba-47f2-8356-b7cb29180f44" />


### NULL VALUES: 
<img width="1769" height="426" alt="image" src="https://github.com/user-attachments/assets/348e6c3f-7dea-4f57-9450-65e1932a073c" />


### NORMALIZED DATA:
<img width="1144" height="370" alt="image" src="https://github.com/user-attachments/assets/35a037b1-ea4c-4d0d-9c7f-482185ae6378" />

### DATA SPLITTING:
<img width="1733" height="406" alt="image" src="https://github.com/user-attachments/assets/922aa317-fd29-4a59-ad64-71e718fda5ea" />
<img width="961" height="495" alt="image" src="https://github.com/user-attachments/assets/942be01e-38af-45e7-b952-36d88632c403" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python using a data set downloaded from Kaggle.


