# EX-06-Feature-Transformation

# AIM:

To Perform the various feature transformation techniques on a dataset and save the data to a file.

# Explanation:

Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:

STEP 1:

Read the given Data

STEP 2:

Clean the Data Set using Data Cleaning Process

STEP 3:

Apply Feature Transformation techniques to all the feature of the data set

STEP 4:

Save the data to the file.

# CODE:
```
Program Developed: D.vishnu vardhan reddy
Register number:212221230023
```
# Data_to_Transform.csv :
```
#Importing packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal')

df=pd.read_csv("Data_to_Transform.csv")
df

#checking and analysing the data
df.isnull().sum()

#checking for skewness of data
df.skew()

#applying data transformations
dfmp=pd.DataFrame()
#for Moderate Positive Skew
#function transformation
dfmp["Moderate Positive Skew"]=df["Moderate Positive Skew"]
dfmp["MPS_log"]=np.log(df["Moderate Positive Skew"]) 
dfmp["MPS_rp"]=np.reciprocal(df["Moderate Positive Skew"])
dfmp["MPS_sqr"]=np.sqrt(df["Moderate Positive Skew"])
#power transformation
dfmp["MPS_yj"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])
dfmp["MPS_bc"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
#quantile transformation
dfmp["MPS_qt"]=qt.fit_transform(df[["Moderate Positive Skew"]])
dfmp.skew()

dfmp.drop('MPS_rp',axis=1,inplace=True)
dfmp.skew()

dfmp

#for Highly Positive Skew
#function transformation
dfhp=pd.DataFrame()
dfhp["Highly Positive Skew"]=df["Highly Positive Skew"]
dfhp["HPS_log"]=np.log(df["Highly Positive Skew"]) 
dfhp["HPS_rp"]=np.reciprocal(df["Highly Positive Skew"])
dfhp["HPS_sqr"]=np.sqrt(df["Highly Positive Skew"])
#power transformation
dfhp["HPS_yj"], parameters=stats.yeojohnson(df["Highly Positive Skew"])
dfhp["HPS_bc"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
#quantile transformation
dfhp["HPS_qt"]=qt.fit_transform(df[["Highly Positive Skew"]])
dfhp.skew()

dfhp.drop('HPS_sqr',axis=1,inplace=True)
dfhp.skew()

dfhp

#for Moderate Negative Skew
dfmn=pd.DataFrame()
#function transformation
dfmn["Moderate Negative Skew"]=df["Moderate Negative Skew"]
dfmn["MNS_rp"]=np.reciprocal(df["Moderate Negative Skew"])
dfmn["MNS_sq"]=np.square(df["Moderate Negative Skew"])
#power transformation
dfmn["MNS_yj"], parameters=stats.yeojohnson(df["Moderate Negative Skew"]) 
#quantile transformation
dfmn["MNS_qt"]=qt.fit_transform(df[["Moderate Negative Skew"]])
dfmn.skew()

dfmn.drop('MNS_rp',axis=1,inplace=True)
dfmn.skew()

dfmn

#for Highly Negative Skew
dfhn=pd.DataFrame()
#function transformation
dfhn["Highly Negative Skew"]=df["Highly Negative Skew"]
dfhn["HNS_rp"]=np.reciprocal(df["Highly Negative Skew"])
dfhn["HNS_sq"]=np.square(df["Highly Negative Skew"])
#phwer transformation
dfhn["HNS_yj"], parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
#quantile transformation
dfhn["HNS_qt"]=qt.fit_transform(df[["Highly Negative Skew"]])
dfhn.skew()

dfhn.drop('HNS_rp',axis=1,inplace=True)
dfhn.skew()

dfhn

#graphical representation
#for Moderate Positive Skew
df["Moderate Positive Skew"].hist()
dfmp["MPS_log"].hist()
dfmp["MPS_sqr"].hist()
dfmp["MPS_bc"].hist()
dfmp["MPS_yj"].hist()
sm.qqplot(df['Moderate Positive Skew'],line='45')
plt.show()
sm.qqplot(dfmp['MPS_qt'],line='45')
plt.show()

#for Highly Positive Skew
df["Highly Positive Skew"].hist()
dfhp["HPS_log"].hist()
dfhp["HPS_rp"].hist()
dfhp["HPS_bc"].hist()
dfhp["HPS_yj"].hist()
sm.qqplot(df['Highly Positive Skew'],line='45')
plt.show()
sm.qqplot(dfhp['HPS_qt'],line='45')
plt.show()

#for Moderate Negative Skew
df["Moderate Negative Skew"].hist()
dfmn["MNS_sq"].hist()
dfmn["MNS_yj"].hist()
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
sm.qqplot(dfmn['MNS_qt'],line='45')
plt.show()

# for Highly Negative Skew
df["Highly Negative Skew"].hist()
dfhn["HNS_sq"].hist()
dfhn["HNS_yj"].hist()
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
sm.qqplot(dfhn['HNS_qt'],line='45')
plt.show()
```
# OUTPUT :
![v](https://user-images.githubusercontent.com/94175324/168818224-06c5e5ed-1e58-4454-9652-b57c53cc0ccd.png)
![v1](https://user-images.githubusercontent.com/94175324/168818268-25f006f9-86ac-47e2-a8be-f099a089b56a.png)
![v2](https://user-images.githubusercontent.com/94175324/168818297-0e76442b-d33a-495d-a8de-5da55fa15576.png)
![v3](https://user-images.githubusercontent.com/94175324/168818336-28a0fb7b-729b-4887-b273-900cc7dc44bc.png)
![v4](https://user-images.githubusercontent.com/94175324/168818373-54b351e4-84fc-4c98-93ef-8b5cae74e40c.png)
![v5](https://user-images.githubusercontent.com/94175324/168818409-3ae51a41-198d-47be-98db-820fc6e3c0da.png)
![v6](https://user-images.githubusercontent.com/94175324/168818450-7eaf2975-f0bd-4e86-ad38-ff98b1722b7e.png)
![v7](https://user-images.githubusercontent.com/94175324/168818513-983b6611-a98a-442c-a9cc-00a0ee47d5e0.png)
![v8](https://user-images.githubusercontent.com/94175324/168818552-488a7c4a-f17c-42be-bc5e-e07b18bd89e6.png)
![v9](https://user-images.githubusercontent.com/94175324/168818577-c8778f35-c080-4641-86fc-456127ed98af.png)
![v10](https://user-images.githubusercontent.com/94175324/168818602-7579bb6b-1935-42f7-8689-e4749f2c7859.png)
![v11](https://user-images.githubusercontent.com/94175324/168818625-60045d48-f12e-43f7-89a9-1d8386c66832.png)
![v12](https://user-images.githubusercontent.com/94175324/168818651-4b22f09c-c37a-416e-877d-aff9b8e47e2e.png)
![v13](https://user-images.githubusercontent.com/94175324/168818691-dd934a33-6379-48c7-ad79-45a7beffaa7b.png)
![v14](https://user-images.githubusercontent.com/94175324/168818712-db438c8f-994c-450b-a405-7b54ab9b9769.png)
![v15](https://user-images.githubusercontent.com/94175324/168818743-2a5aa0d8-e35f-408b-a255-b183bfee268b.png)
![v16](https://user-images.githubusercontent.com/94175324/168818768-c3ecb707-d269-44c4-9452-8436c274784b.png)
![v17](https://user-images.githubusercontent.com/94175324/168818796-e2df5131-661d-4aa2-8a6a-a3717aca747c.png)
![v18](https://user-images.githubusercontent.com/94175324/168818830-c65dce4b-0592-474c-b36a-c0dfeba277e9.png)
![v19](https://user-images.githubusercontent.com/94175324/168818865-e3c8dee7-fc88-4af8-855f-f12e70ffa4da.png)
![v20](https://user-images.githubusercontent.com/94175324/168818917-3d96853f-2725-4dd0-b413-1e687fde274b.png)
![v21](https://user-images.githubusercontent.com/94175324/168818956-c7d1816e-1167-4521-8d9e-acb9f909c016.png)
![v22](https://user-images.githubusercontent.com/94175324/168818985-c1f228fd-616f-4943-ad09-198715abac2c.png)
![v23](https://user-images.githubusercontent.com/94175324/168819012-3bc7bca8-08b5-4fc2-8993-82cf39ceaf0e.png)
![v24](https://user-images.githubusercontent.com/94175324/168819048-801abe30-4eb0-4bf6-8999-cc200818e811.png)




# For Titanic_dataset.csv:
```
# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df=pd.read_csv("titanic_dataset.csv")
df

#checking and analysing the data
df.isnull().sum()
# cleaning data
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
#encoding categorical data
from sklearn.preprocessing import OrdinalEncoder
embark=["C","S","Q"]
emb=OrdinalEncoder(categories=[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])

from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
df.skew()

#feature in 0.5 to -0.5 range: Embarked Survived Age (A,S pos)(E neg)
#features that are in skew
#neg:Pclass 
#pos:Sex SibSp Parch Fare 
df["Age_1"]=qt.fit_transform(df[["Age"]])
df["Survived_1"]=qt.fit_transform(df[["Survived"]])
df["Embarked_1"]=qt.fit_transform(df[["Embarked"]])
df["Pclass_sq"]=np.square(df["Pclass"])
df["Pclass_qt"]=qt.fit_transform(df[["Pclass"]])
df["SibSp_yj"], parameters=stats.yeojohnson(df["SibSp"])
df["SibSp_qt"]=qt.fit_transform(df[["SibSp"]])

df["Parch_yj"], parameters=stats.yeojohnson(df["Parch"])
df["Parch_qt"]=qt.fit_transform(df[["Parch"]])

df["Fare_yj"], parameters=stats.yeojohnson(df["Fare"])
df["Fare_qt"]=qt.fit_transform(df[["Fare"]])

df["Sex_yj"], parameters=stats.yeojohnson(df["Sex"])
df["Sex_qt"]=qt.fit_transform(df[["Sex"]])
df.skew()

#taking closer to range skew values
df.drop('Sex_yj',axis=1,inplace=True)
df.drop('Pclass_qt',axis=1,inplace=True)
df.drop('SibSp_qt',axis=1,inplace=True)
df.drop('Parch_qt',axis=1,inplace=True)
df.drop('Fare_qt',axis=1,inplace=True)
df.skew()

#graph representation
df["Sex"].hist()
df["Sex_qt"].hist()
df["SibSp"].hist()
df["SibSp_yj"].hist()
df["Parch"].hist()
df["Parch_yj"].hist()
df["Fare"].hist()
df["Fare_yj"].hist()
df["Pclass"].hist()
df["Pclass_sq"].hist()
```
# OUTPUT:
![e](https://user-images.githubusercontent.com/94175324/168820783-7037caef-c4ae-4f89-92fb-bd9180304dd4.png)
![e2](https://user-images.githubusercontent.com/94175324/168820859-8b04988b-2959-463b-8959-b24dca8ba72c.png)
![e3](https://user-images.githubusercontent.com/94175324/168820895-92d70cc4-9880-430c-bf2c-e73499929a10.png)
![e4](https://user-images.githubusercontent.com/94175324/168820924-7daac5b4-3680-4715-9bf2-23f27fe59aec.png)
![e5](https://user-images.githubusercontent.com/94175324/168820941-b13b0d74-69f6-45a3-ab6b-04b9eed3bd1a.png)
![e6](https://user-images.githubusercontent.com/94175324/168820973-d21ed005-cc92-40c9-a36b-6352f93a97ba.png)
![e7](https://user-images.githubusercontent.com/94175324/168820992-6fee831e-d555-48f9-9505-71248c5ecf5f.png)
![e8](https://user-images.githubusercontent.com/94175324/168821034-26c1e68e-fbdc-4512-ba6b-8054c6bb7fe0.png)
![e9](https://user-images.githubusercontent.com/94175324/168821077-8754c366-5f33-422b-b298-2a9cd14d3df9.png)
![e10](https://user-images.githubusercontent.com/94175324/168821096-05e9935c-d4df-4acc-877b-c0a2e6d2e07a.png)
![e11](https://user-images.githubusercontent.com/94175324/168821137-f347eae1-070c-409e-8f81-e5209c66da26.png)
![e12](https://user-images.githubusercontent.com/94175324/168821167-d4c18bee-84ae-4f7b-9615-f5dec451ea5c.png)




# RESULT :
The various feature transformation techniques has been performed on the given datasets and the data are saved to a file.

