#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #collection of command style functions that make matplotlib work like MATLAB & data visulization
import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib.

df = pd.read_csv('C:/Users/king/Desktop/jupyterythonassignment/Python_Assignment10/Cohorts/Most-Recent-Cohorts-Institution.csv') # Get the data
df.head(10)


# In[22]:


# Summary statistics of numerical columns
print("\nSummary Statistics:")
print(df.describe())


# In[86]:


import pandas as pd

def data_information(df):
    count = df.count()
    unique = df.nunique()
    duplicated = count - unique
    missing = df.isnull().sum()
    typedt = df.dtypes
    column_info_dict = {
        "Non-Nulls": count,
        "Unique Values": unique,
        "Duplicated": duplicated,
        "Missing Values": missing,
        "Column Type": typedt,
    }
    data_information = pd.DataFrame(column_info_dict).style.background_gradient()
    return data_information

# 'df' is your DataFrame
data_info = data_information(df)
data_info


# In[26]:


print ('number of duplicated rows in the dataset : ' ,  df.duplicated().sum())
print ('Number of Rows and columns in the dataset : ' , df.shape)


# In[27]:


cat1= df.nunique()[df.nunique()==2]
cat1


# In[28]:


df.drop(columns= cat1.index,inplace= True)


# In[29]:


print ('Number of Rows and columns in the dataset : ' , df.shape)


# In[30]:


df.nunique()[df.nunique()<=20]


# In[31]:


cat2= df.nunique()[df.nunique()==3]
cat2


# In[32]:


df.drop(columns= cat2.index, inplace= True)


# In[33]:


df.shape


# In[34]:


df.nunique()[df.nunique()<=20]


# In[35]:


df.REGION.unique()


# In[36]:


df.REGION.isnull().sum()


# In[37]:


df.LOCALE.unique()


# In[39]:


# Which colleges are in the range of a SAT score
sat1400 = df.query('SAT_AVG > 1400 & SAT_AVG < 1550')
#print (sat1400['CITY'])
# Which colleges are in Boston area
#bsat1400 = sat1400[sat1400['City'] == 'Boston']
#print (wsat1400['INSTNM'])
import seaborn as sns
corrmat = sat1400.corr()
sns.heatmap(corrmat, 
            xticklabels=corrmat.columns.values,
           yticklabels=corrmat.columns.values)


# In[81]:



df1 = df[['UNITID'  , 'INSTNM'  , 'PREDDEG' , 'CONTROL' , 'SATVRMID' , 'SATMTMID'  ,'ACTCMMID','ACTENMID' , 'ACTMTMID' ]]
df1


# In[88]:


#Create new predection data fram
dfp = df[['NPT4_PUB' , 'NPT4_PRIV']]
#sum the price for public and private colleges
dfp.NPT4_PUB =  dfp.NPT4_PUB.fillna(0)
dfp.NPT4_PRIV =  dfp.NPT4_PRIV.fillna(0)
dfp['Net_Price'] = dfp.NPT4_PUB + dfp.NPT4_PRIV
df['Net_Price'] = dfp['Net_Price']
dfp.Net_Price = dfp.Net_Price .apply (lambda x: np.nan if x<1 else x)
dfp.Net_Price = dfp.Net_Price .apply (lambda x: np.nan if x>55000 else x)


# In[89]:


fig , ax = plt.subplots(figsize = (8,8))
dfp.Net_Price.hist()
plt.title ('Histogram for the Destribution of Net_Price - (Y-target)' , fontsize = 14 , fontfamily='serif');


# In[66]:


df2 = df.groupby('PREDDEG')['Net_Price'].mean().reset_index().rename({'Net_Price':'Average_Net_Price'} , axis =1)
df2


# In[109]:


fig = plt.figure(figsize=(22,13) , facecolor="#fbfbfb")
x = [2.5,3.6,4.8,5.7]
y = [4,6,5,3]
width = [ 0.8, 1.2, 1.0, 0.6]
color = ['#D4A492','#E0B30E','#8D918A','#D0D9C3']

s_num = ['3rd','1st','2nd','4th']

fontsize= [ 45, 60, 40, 25]
x_num = [2.4,3.45,4.65,5.7]
x_char = [2.62,3.7,4.9,5.78]
y_char = [4,5.95,5,3.05]
alpha = [ 0.6, 1, 0.8, 0.3]
s = ['Certificate','Bachelor','Not classified','Associate']
s_position = [ 1.8, 3, 2.5, 1.5]

for i in range(4):
    plt.bar(x=x[i],height=y[i],width=width[i],color=color[i],alpha=alpha[i])
    plt.text(s=s[i],x=x[i],y=s_position[i],va='bottom',ha='center',fontsize=fontsize[i],alpha=alpha[i])
    plt.text(s=s_num[i],x=x_num[i],y=y[i]+0.1,va='bottom',ha='center',fontsize=fontsize[i],alpha=alpha[i])
    #plt.text(s=s_char[i],x=x_char[i],y=y_char[i],va='bottom',ha='center',fontsize=fontsize[i]-25,alpha=alpha[i])
    
plt.text(2,8,'Ranking of Degree Granting across the Average Net-Price' , fontsize = 22  , fontweight='bold', 
         fontfamily='serif')
plt.text(2,7.5,'Average net price for colleges granting Associate degree was 13668 usd' , fontsize = 16  ,  
         fontfamily='serif')
plt.text(2,7.1,'While the Average net price for colleges granting Bachelor degree was 19054 usd' , fontsize = 16  ,  
         fontfamily='serif')

plt.axis('off')
plt.show()


# In[95]:


columns = ['STABBR' ,'PREDDEG', 'LOCALE', 'CONTROL', 'HBCU', 'PBI',
           'ANNHI', 'TRIBAL', 'AANAPII', 'HSI', 'NANTI','SATVRMID', 'SATMTMID' ,'ACTCMMID', 'ACTENMID', 
           'ACTMTMID'  , 'SAT_AVG','SAT_AVG_ALL'  , 'MD_EARN_WNE_P10' , 'GT_25K_P6' , 'GRAD_DEBT_MDN_SUPP'
           , 'GRAD_DEBT_MDN10YR_SUPP' , 'RPY_3YR_RT_SUPP']
dfp[columns] = df[columns]
dfp.head()


# Predictive Model

# In[44]:


dfp.Net_Price.hist()
plt.title ('Histogram for the Destribution of Net_Price - (Y-target)' , fontsize = 14 , fontfamily='serif');


# In[96]:


#drop the private and puplic net price columns
dfp.drop(['NPT4_PUB' , 'NPT4_PRIV'] , axis = 1 , inplace = True)


#1113 collage had no information about the net price , will reduce the data frame with the non-values only 
dfp =dfp[dfp['Net_Price'].notnull()]


#Remove the 'PrivacySuppressed' and replace it with nan values

#dfp.MD_EARN_WNE_P10 = dfp.MD_EARN_WNE_P10.apply(lambda x: np.nan if x =='PrivacySuppressed' else x)
dfp.GT_25K_P6 = dfp.GT_25K_P6.apply(lambda x: np.nan if x =='PrivacySuppressed' else x)
dfp.GRAD_DEBT_MDN_SUPP = dfp.GRAD_DEBT_MDN_SUPP.apply(lambda x: np.nan if x =='PrivacySuppressed' else x)
dfp.GRAD_DEBT_MDN10YR_SUPP = dfp.GRAD_DEBT_MDN10YR_SUPP.apply(lambda x: np.nan if x =='PrivacySuppressed' else x)
dfp.RPY_3YR_RT_SUPP = dfp.RPY_3YR_RT_SUPP.apply(lambda x: np.nan if x =='PrivacySuppressed' else x)



# Convert columns to float

dfp['GT_25K_P6'] = pd.to_numeric(dfp['GT_25K_P6'])
dfp['MD_EARN_WNE_P10'] = pd.to_numeric(dfp['MD_EARN_WNE_P10'])
dfp['GRAD_DEBT_MDN_SUPP'] = pd.to_numeric(dfp['GRAD_DEBT_MDN_SUPP'])
dfp['GRAD_DEBT_MDN10YR_SUPP'] = pd.to_numeric(dfp['GRAD_DEBT_MDN10YR_SUPP'])
dfp['RPY_3YR_RT_SUPP'] = pd.to_numeric(dfp['RPY_3YR_RT_SUPP'])


#Label encoder for the STABBR
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfp.STABBR = le.fit_transform(dfp.STABBR)

#MD_EARN_WNE_P10 , remove outlayers and replace the nan with the mean
dfp.MD_EARN_WNE_P10 = dfp.MD_EARN_WNE_P10.apply(lambda x : np.nan if x>80000 else x)
dfp.MD_EARN_WNE_P10.fillna(dfp.MD_EARN_WNE_P10.mean() , inplace = True)

#GT_25K_P6 , fill nan with the mean
dfp.GT_25K_P6.fillna(dfp.GT_25K_P6.mean() , inplace  = True)

#GRAD_DEBT_MDN_SUPP remove the outlayers and fill the nan vlaues with the mean
dfp.GRAD_DEBT_MDN_SUPP = dfp.GRAD_DEBT_MDN_SUPP.apply(lambda x : np.nan if x>40000 else x)
dfp.GRAD_DEBT_MDN_SUPP.fillna(dfp.GRAD_DEBT_MDN_SUPP.mean() , inplace = True)

#GRAD_DEBT_MDN10YR_SUPP remove the outlayers and fill the nan vlaues with the mean
dfp.GRAD_DEBT_MDN10YR_SUPP = dfp.GRAD_DEBT_MDN10YR_SUPP.apply(lambda x : np.nan if x>400 else x)
dfp.GRAD_DEBT_MDN10YR_SUPP.fillna(dfp.GRAD_DEBT_MDN10YR_SUPP.mean() , inplace = True)

#RPY_3YR_RT_SUPP fill nan vlaues with the mean
dfp.RPY_3YR_RT_SUPP.fillna(dfp.RPY_3YR_RT_SUPP.mean() , inplace = True)

#Feature engineering one column for SAT
dfp.SATVRMID.fillna(0 , inplace = True)
dfp.SATMTMID.fillna(0 , inplace = True)
dfp['SAT'] = dfp.SATVRMID +dfp.SATMTMID
dfp.drop(['SATVRMID' , 'SATMTMID'] , axis  = 1  , inplace  = True)


#Feature engineering one column for SAT_Avg
dfp.SAT_AVG.fillna(0 , inplace = True)
dfp.SAT_AVG_ALL.fillna(0 , inplace = True)
dfp['SAT_AVG'] = dfp.SAT_AVG +dfp.SAT_AVG_ALL
dfp.drop(['SAT_AVG' , 'SAT_AVG_ALL'] , axis  = 1  , inplace  = True)


#Feature engineering one column for ACT Score 
dfp.ACTCMMID.fillna(0 , inplace = True)
dfp.ACTENMID.fillna(0 , inplace = True)
dfp.ACTMTMID.fillna(0 , inplace = True)
dfp['ACT_Score'] = dfp.ACTCMMID +dfp.ACTENMID+dfp.ACTMTMID
dfp.drop(['ACTCMMID' , 'ACTENMID' , 'ACTMTMID'] , axis  = 1  , inplace  = True)


# In[97]:


y = dfp.Net_Price.values
X = dfp.drop('Net_Price' , axis = 1).values


# In[98]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, 
                                                    random_state=42)


# In[99]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor()
regr.fit(X_train, y_train)


# In[100]:


train_prediction =  regr.predict(X_train)
test_predection = regr.predict(X_test)


# In[101]:


Fig , ax = plt.subplots(figsize = (15 ,5))
ax.plot(y_train[0:100]  , label = 'Original Net Price')
ax.plot(train_prediction[0:100] , label = 'Prediction Net Price' )
plt.legend()
plt.title('Original VS Predicted Net price Comparasion for the first 100 Colleges in the train dataset' , fontsize=18.5, 
         fontweight='light', 
         fontfamily='serif');


# In[102]:


Fig , ax = plt.subplots(figsize = (15 ,5))
ax.plot(y_test[0:100], label = 'Original Net Price' )
ax.plot(test_predection[0:100], label = 'Prediction Net Price' )
plt.legend()
plt.title('Original VS Predicted Net price Comparasion for the first 100 Colleges in the Test dataset' , fontsize=18.5, 
         fontweight='light', 
         fontfamily='serif');


# In[103]:


from sklearn.metrics import mean_squared_error
print ('Root Mean square error for the train data , ' , 
       round(mean_squared_error(y_train, train_prediction , squared=False) , 2))
print ('Root Mean square error for the train data , ' ,
       round(mean_squared_error(y_test, test_predection , squared=False) , 2))


# In[104]:


final_df = pd.DataFrame.from_dict({"original net price":y_train , "Predicted net price":train_prediction})
final_df = pd.DataFrame.round(final_df)
final_df['Percentage of Error %'] = abs(final_df['original net price']-final_df['Predicted net price'])/final_df['original net price']*100
final_df =final_df.round(1)
final_df.sample(20)


# In[105]:


from sklearn.tree import DecisionTreeRegressor
regr = DecisionTreeRegressor()
regr.fit(X_train, y_train)


# In[111]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

regr = DecisionTreeRegressor()
regr.fit(X_train, y_train)

# Predict the target values using the model
y_pred = regr.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)


# In[ ]:




