#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score, f1_score, auc, balanced_accuracy_score, cohen_kappa_score, log_loss, roc_auc_score
from imblearn.over_sampling import SMOTE
import sys
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
import matplotlib as mpl
import tempfile
import seaborn as sns
import os
import sklearn
from datetime import datetime, timedelta
from collections import Counter
import threading
import multiprocessing

mpl.rcParams['figure.figsize'] = (25, 25)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('max_colwidth', 500)

get_ipython().magic('matplotlib inline')


# ## Load dataset, Outlier treatment
# 
# > Feed in the BU unit name at the end of the function

# In[2]:


def dataset(sub):   
    '''Loads, merges and removes outliers from data. Transforms and creates subset on (input)business segment. 
    The data has to bee in data folder with name OOS_All and feather format'''
    
    # Read both data files and merge
    print("Loading data...")
    #load data from feather of csv form
    df = pd.read_feather('OOS_All.feather')
    dfh = pd.read_excel('Product Heirarchy_v2.xlsx')
    dfn = df.merge(dfh,on='P_ID',how='left')

    # Convert DAY_DT to correct date format, remove outliers, & replace BOH with MDQ for outliers
    print("Removing outliers ...")
    dfn['DAY_DT']= pd.to_datetime(dfn['DAY_DT'])
    dfn = dfn.query('boh<10000 & mdq<1000 & sell_thru<500')
    dfn['boh_replaced'] = np.where(dfn['boh']<0,dfn['mdq'],dfn['boh'])
    dfn.drop(columns=['boh'], inplace=True)
    dfn.rename(columns={"boh_replaced":"boh"},inplace=True)
    
    # Remove low volume business segments 
    print("Removing low volume business segments...")
    dfn = dfn.query('MJR_BUS_SEG_NM_TX not in ("BEAUTY CARE","BULK WATER AND COFFEE","BABY CONSUMABLES","OTC HEALTH CARE","PETS")')
    new_df = dfn[~dfn["P_NM"].str.contains("GRAPE")]
    
    #Subset data on input business segment
    print("Subsetting data on",sub,"...")
    sf_df = new_df.query('MJR_BUS_SEG_NM_TX =="'+str(sub)+'"')
    #sf_df = new_df.query('P_ID =="'+str(p)+'"')
    
    sf_df['y']= np.where(sf_df['oos_tn']==0,0,1)
    sf_df['DAY_DT_Ordinal'] = sf_df['DAY_DT'].apply(datetime.toordinal)
    sf_df.drop(columns = ['MJR_MDS_ARE_NM_TX','MJR_BUS_SEG_NM_TX','MJR_P_CT_NM_TX', 'MJR_P_SUB_CT_ID', 'MJR_P_CLS_NM_TX','P_NM','MJR_MDS_ARE_ID', 'MJR_BUS_SEG_ID', 'MJR_P_CT_ID','MJR_P_CLS_ID'],inplace=True)
    
    return sf_df

###Input segment to be modelled 

bkp = dataset('MEAT')
df = bkp.copy()


# ## Correlation PLOT
# > input variables

# In[43]:


df1 = df[['P_ID','UT_ID','PROMO_FLG','DP_fcst','DP_units','DP_OOS','DP_dgt_tn','dp_oos_chain','mdq','boh','oos_tn']]


# In[44]:


corr = df1.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ## Feature Engineering
# 
# > Load the feature py file
# > Engineer 31 features for L3 level

# In[6]:


import features

df = features.create_date_features(df)


# ## Filter data using Business Logic
# 
# > 1. GROCERY_DSD - OOS_percentage > 4 & oos_count>100
# > 2. DRY_GROCERY - OOS_percentage > 4 & oos_count>100
# > 3. PRODUCE - OOS_percentage > 5 & oos_count>100
# > 4. DAIRY - OOS_percentage > 4 & oos_count>100
# > 5. FROZEN FOODS - OOS_percentage > 4 & oos_count>100

# In[7]:


#OOS rate at L3 level
def OOS_Rate_L3(df):
    temp1 = df.MJR_P_SUB_CT_NM_TX.value_counts().reset_index()
    temp1.rename(columns={"index": "L3","MJR_P_SUB_CT_NM_TX":'countofrecords'}, inplace=True)
    temp2 = df.groupby('MJR_P_SUB_CT_NM_TX').y.sum().reset_index()
    temp2.rename(columns={"MJR_P_SUB_CT_NM_TX": "L3","y":'countofoos'}, inplace=True)
    temp3 = temp1.merge(temp2, left_on='L3', right_on='L3')
    temp3['OOS_Rate_L3'] = (temp3.countofoos/(temp3.countofrecords))*100
    del temp1, temp2
    return temp3


# In[8]:


#OOS rate at product level
def OOS_Product_Level(df):
    df['oos_tn_1']= np.where(df['oos_tn']==0,0,1)
    temp = df[['P_ID','oos_tn_1']].groupby('P_ID').count().reset_index()
    temp.rename(columns={"oos_tn_1": "total_records"}, inplace=True)
    temp2 = df[['P_ID','oos_tn_1']].groupby('P_ID').sum().reset_index()
    temp2.rename(columns={"oos_tn_1": "oos_count"}, inplace=True)
    temp3 = temp.merge(temp2, left_on='P_ID', right_on='P_ID')
    temp3['OOS_percentage'] = (temp3.oos_count/(temp3.total_records))*100
    df.drop(columns='oos_tn_1',inplace=True)
    del temp, temp2
    return temp3


# In[9]:


OOS_Rate_L3_BU = OOS_Rate_L3(df)


# ## Split L5 to L3 level after applying Business logic
# 
# > 1. Create a dict of dataframes on L3 level
# > 2. Spilt into list to process using power of multiprocessing
# 

# In[24]:


def L3_level_dataframes(df):
#create unique list of names
    L3 = df.MJR_P_SUB_CT_NM_TX.unique()
    
    print('Number of L3 level in BU:',len(L3)) 

    #create a data frame dictionary to store your data frames
    BU = {elem : pd.DataFrame for elem in L3}

    for key in BU.keys():
        BU[key] = df[:][df.MJR_P_SUB_CT_NM_TX == key]
    
    df_list_full = []
    for key in BU.keys():
        df_list_full.append((BU[key]))
       
    return df_list_full

BU_LIST_L3 = L3_level_dataframes(df)


# In[28]:


#break into multiple lists

df_list = BU_LIST_L3[0:2].copy()
df_list1 = BU_LIST_L3[2:4].copy()
df_list2 = BU_LIST_L3[4:7].copy()


# In[31]:


#check number of L3 levels in all lists
print('Number of L3 in BU in all lists:',len(df_list)+len(df_list1)+len(df_list2))


# ## Using multiprocessing to create new feature

# In[32]:


p = multiprocessing.Pool(processes=3)
df_list = p.map(features.create_day_mean_units, df_list)
df_list = p.map(features.create_day_mean_sales, df_list)
df_list = p.map(features.create_day_mean_chain_oos, df_list)
df_list = p.map(features.create_day_mean_dgt_txn_occ, df_list)
df_list = p.map(features.create_day_sd_units, df_list)
df_list = p.map(features.create_day_sd_sales, df_list)
df_list = p.map(features.create_day_sd_chain_oos, df_list)
df_list = p.map(features.create_day_sd_dgt_txn_occ, df_list)

df_list = p.map(features.create_week_num_mean_units, df_list)
df_list = p.map(features.create_week_num_mean_sales, df_list)
df_list = p.map(features.create_week_num_mean_chain_oos, df_list)
df_list = p.map(features.create_week_num_mean_dgt_txn_occ, df_list)
df_list = p.map(features.create_week_num_sd_units, df_list)
df_list = p.map(features.create_week_num_sd_sales, df_list)
df_list = p.map(features.create_week_num_sd_chain_oos, df_list)
df_list = p.map(features.create_week_num_sd_dgt_txn_occ, df_list)

df_list = p.map(features.create_month_mean_units, df_list)
df_list = p.map(features.create_month_mean_sales, df_list)
df_list = p.map(features.create_month_mean_chain_oos, df_list)
df_list = p.map(features.create_month_mean_dgt_txn_occ, df_list)
df_list = p.map(features.create_month_sd_units, df_list)
df_list = p.map(features.create_month_sd_sales, df_list)
df_list = p.map(features.create_month_sd_chain_oos, df_list)
df_list = p.map(features.create_month_sd_dgt_txn_occ, df_list)

df_list = p.map(features.create_quarter_mean_units, df_list)
df_list = p.map(features.create_quarter_mean_sales, df_list)
df_list = p.map(features.create_quarter_mean_chain_oos, df_list)
df_list = p.map(features.create_quarter_mean_dgt_txn_occ, df_list)
df_list = p.map(features.create_quarter_sd_units, df_list)
df_list = p.map(features.create_quarter_sd_sales, df_list)
df_list = p.map(features.create_quarter_sd_chain_oos, df_list)
df_list = p.map(features.create_quarter_sd_dgt_txn_occ, df_list)
p.close()


# In[33]:


p = multiprocessing.Pool(processes=3)
df_list1 = p.map(features.create_day_mean_units, df_list1)
df_list1 = p.map(features.create_day_mean_sales, df_list1)
df_list1 = p.map(features.create_day_mean_chain_oos, df_list1)
df_list1 = p.map(features.create_day_mean_dgt_txn_occ, df_list1)
df_list1 = p.map(features.create_day_sd_units, df_list1)
df_list1 = p.map(features.create_day_sd_sales, df_list1)
df_list1 = p.map(features.create_day_sd_chain_oos, df_list1)
df_list1 = p.map(features.create_day_sd_dgt_txn_occ, df_list1)

df_list1 = p.map(features.create_week_num_mean_units, df_list1)
df_list1 = p.map(features.create_week_num_mean_sales, df_list1)
df_list1 = p.map(features.create_week_num_mean_chain_oos, df_list1)
df_list1 = p.map(features.create_week_num_mean_dgt_txn_occ, df_list1)
df_list1 = p.map(features.create_week_num_sd_units, df_list1)
df_list1 = p.map(features.create_week_num_sd_sales, df_list1)
df_list1 = p.map(features.create_week_num_sd_chain_oos, df_list1)
df_list1 = p.map(features.create_week_num_sd_dgt_txn_occ, df_list1)

df_list1 = p.map(features.create_month_mean_units, df_list1)
df_list1 = p.map(features.create_month_mean_sales, df_list1)
df_list1 = p.map(features.create_month_mean_chain_oos, df_list1)
df_list1 = p.map(features.create_month_mean_dgt_txn_occ, df_list1)
df_list1 = p.map(features.create_month_sd_units, df_list1)
df_list1 = p.map(features.create_month_sd_sales, df_list1)
df_list1 = p.map(features.create_month_sd_chain_oos, df_list1)
df_list1 = p.map(features.create_month_sd_dgt_txn_occ, df_list1)

df_list1 = p.map(features.create_quarter_mean_units, df_list1)
df_list1 = p.map(features.create_quarter_mean_sales, df_list1)
df_list1 = p.map(features.create_quarter_mean_chain_oos, df_list1)
df_list1 = p.map(features.create_quarter_mean_dgt_txn_occ, df_list1)
df_list1 = p.map(features.create_quarter_sd_units, df_list1)
df_list1 = p.map(features.create_quarter_sd_sales, df_list1)
df_list1 = p.map(features.create_quarter_sd_chain_oos, df_list1)
df_list1 = p.map(features.create_quarter_sd_dgt_txn_occ, df_list1)
p.close()


# In[35]:


p = multiprocessing.Pool(processes=3)
df_list2 = p.map(features.create_day_mean_units, df_list2)
df_list2 = p.map(features.create_day_mean_sales, df_list2)
df_list2 = p.map(features.create_day_mean_chain_oos, df_list2)
df_list2 = p.map(features.create_day_mean_dgt_txn_occ, df_list2)
df_list2 = p.map(features.create_day_sd_units, df_list2)
df_list2 = p.map(features.create_day_sd_sales, df_list2)
df_list2 = p.map(features.create_day_sd_chain_oos, df_list2)
df_list2 = p.map(features.create_day_sd_dgt_txn_occ, df_list2)

df_list2 = p.map(features.create_week_num_mean_units, df_list2)
df_list2 = p.map(features.create_week_num_mean_sales, df_list2)
df_list2 = p.map(features.create_week_num_mean_chain_oos, df_list2)
df_list2 = p.map(features.create_week_num_mean_dgt_txn_occ, df_list2)
df_list2 = p.map(features.create_week_num_sd_units, df_list2)
df_list2 = p.map(features.create_week_num_sd_sales, df_list2)
df_list2 = p.map(features.create_week_num_sd_chain_oos, df_list2)
df_list2 = p.map(features.create_week_num_sd_dgt_txn_occ, df_list2)

df_list2 = p.map(features.create_month_mean_units, df_list2)
df_list2 = p.map(features.create_month_mean_sales, df_list2)
df_list2 = p.map(features.create_month_mean_chain_oos, df_list2)
df_list2 = p.map(features.create_month_mean_dgt_txn_occ, df_list2)
df_list2 = p.map(features.create_month_sd_units, df_list2)
df_list2 = p.map(features.create_month_sd_sales, df_list2)
df_list2 = p.map(features.create_month_sd_chain_oos, df_list2)
df_list2 = p.map(features.create_month_sd_dgt_txn_occ, df_list2)

df_list2 = p.map(features.create_quarter_mean_units, df_list2)
df_list2 = p.map(features.create_quarter_mean_sales, df_list2)
df_list2 = p.map(features.create_quarter_mean_chain_oos, df_list2)
df_list2 = p.map(features.create_quarter_mean_dgt_txn_occ, df_list2)
df_list2 = p.map(features.create_quarter_sd_units, df_list2)
df_list2 = p.map(features.create_quarter_sd_sales, df_list2)
df_list2 = p.map(features.create_quarter_sd_chain_oos, df_list2)
df_list2 = p.map(features.create_quarter_sd_dgt_txn_occ, df_list2)
p.close()


# ## Create OOS streak feature

# In[87]:


p = multiprocessing.Pool(processes=20)
df_list = p.map(features.not_dp_oos_since_assist, df_list)
p.close()


# In[88]:


df_list[0]


# In[37]:


p = multiprocessing.Pool(processes=10)
df_list1 = p.map(features.not_dp_oos_since_assist, df_list1)
p.close()


# In[89]:


df_list1[0]


# In[38]:


p = multiprocessing.Pool(processes=10)
df_list2 = p.map(features.not_dp_oos_since_assist, df_list2)
p.close()


# In[90]:


df_list2[0]


# In[91]:


## join the L# segments back to BU

Group1 = pd.concat(df_list,axis=0)
Group2 = pd.concat(df_list1,axis=0)
Group3 = pd.concat(df_list2,axis=0)


# In[92]:


df_BU = pd.concat([Group1,Group2,Group3],axis=0).reset_index()  #.to_feather('DELI_with_feature.feather')


# In[93]:


df_BU.drop(columns='index',inplace=True)


# In[95]:


df_BU.isnull().sum()


# ## Prepare data for modelling

# In[97]:


def dataprepformodelling(df):
    modeling_df = df.copy()
    modeling_df = pd.concat([modeling_df, pd.get_dummies(modeling_df['day_of_week'])], axis=1)
    modeling_df = pd.concat([modeling_df, pd.get_dummies(modeling_df['P_ID'], prefix="P_", prefix_sep='_')], axis=1)
    modeling_df = pd.concat([modeling_df, pd.get_dummies(modeling_df['UT_ID'], prefix="UT", prefix_sep='_')], axis=1)
    modeling_df = pd.concat([modeling_df, pd.get_dummies(modeling_df['quarter'], prefix="Q", prefix_sep='')], axis=1)
    modeling_df = pd.concat([modeling_df, pd.get_dummies(modeling_df['month'])], axis=1)
    modeling_df = pd.concat([modeling_df, pd.get_dummies(modeling_df['week_of_year'], prefix="week", prefix_sep='_')], axis=1)
    modeling_df.pop('y')
    modeling_df['y']= np.where(modeling_df['oos_tn']==0,0,1)
    neg, pos = np.bincount(modeling_df['y'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    temp_df = modeling_df.copy()
    col_list = temp_df.columns
    remove_col_list =['dgt_txn_occ','DAY_DT','P_ID','UT_ID','fcst','units',
                  'sales','chain_oos','ds_ls_units','sell_thru','P_NM',
                  'day_of_week','oos_tn','month','quarter','week_of_year',
                  'DP_CHAIN_OOS_INV','DP_OOS_INV','day_of_week','day_of_year','MJR_P_SUB_CT_NM_TX','week_mean_sales',
                  'week_num_sd_sales','month_mean_sales','month_sd_sales','quarter_mean_sales','quarter_sd_sales',
                      'quarter_mean_chain_oos','quarter_sd_units','month_mean_chain_oos','week_mean_chain_oos',
                      'week_mean_units','day_mean_units','quarter_mean_units','month_mean_units',
                      'day_mean_units','day_sd_sales','day_sd_chain_oos','week_mean_units',
    'month_mean_units',
    'month_sd_chain_oos',
    'quarter_mean_units','quarter_sd_chain_oos']
    list_of_cols = list((Counter(col_list) - Counter(remove_col_list)).elements())
    cleaned_df = temp_df[list_of_cols]
    return cleaned_df


# In[98]:


cleaned_df = dataprepformodelling(df_BU)


# In[99]:


#Checking nulls
cleaned_df.isnull().sum()


# ## RF modelling function
# 
# > 1. Split into 75-25
# > 2. Upsampling using SMOTE in 50-50 ratio
# > 3. Min Max scaler
# > 4. Model fitting
# > 5. Results gathering

# In[102]:


def result_matrix(data):
    
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    from datetime import datetime
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler 
    
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier, Perceptron
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, accuracy_score, f1_score, auc, balanced_accuracy_score, cohen_kappa_score, log_loss, roc_auc_score
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB, GaussianNB
    from matplotlib import colors
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import VotingClassifier
    from sklearn.feature_selection import SelectFromModel
    from joblib import dump, load
    
    import time
    df = data
    #print(df.head())

    classifier_list = [RandomForestClassifier(random_state = 0)]
    
    x = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    # Split Data
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)

    #Upsampling
    
    sm = SMOTE(random_state=12, sampling_strategy = 1.0)
    xtrain, ytrain = sm.fit_sample(xtrain, ytrain)
    sc_x = MinMaxScaler() 

    xtrain = sc_x.fit_transform(xtrain)  
    xtest = sc_x.transform(xtest)
    
    print(sum(ytrain))
    print(len(ytrain))
  
    # Metric Lists
    f_score = []
    acc_list = []
    model_list = []
    bacc_list = []
    cohen_list = []
    ll_list = []
    roc_auc_list = []
    t_list =[]
    f_list = []
    cm_list = []
    pr_list =[]
    rc_list = []
    cv_list = []
    
    # Train
    for clfr in classifier_list:
        print("Running: ",clfr)
        start = time.time()
        #s = cross_val_score(clfr,xtrain,ytrain,scoring='f1', cv=5)
#         name = "model_"+str(start).split('.')[0]+"_"+str(start).split('.')[1]+".joblib"
        clfr.fit(xtrain, ytrain)
        y_pred = (clfr.predict(xtest)) 
        end= time.time()
        
#         dump(clfr, 'Models/'+name)
        #print(name,clfr)

        acc = accuracy_score(ytest, y_pred)
        f = f1_score(ytest, y_pred)
        bacc = balanced_accuracy_score(ytest, y_pred)
        cohen = cohen_kappa_score(ytest, y_pred)
        ll = log_loss(ytest, y_pred)
        roc_auc = roc_auc_score(ytest, y_pred)
        cm = confusion_matrix(ytest,y_pred)
        pr = precision_score(ytest,y_pred)
        rc = recall_score(ytest,y_pred)
        t = end-start
        
        model_list.append(clfr)
        f_score.append(f)
        acc_list.append(acc)
        bacc_list.append(bacc)
        cohen_list.append(cohen)
        ll_list.append(ll)
        roc_auc_list.append(roc_auc)
        t_list.append(t)
        cm_list.append(cm)
        pr_list.append(pr)
        rc_list.append(rc)
        #cv_list.append(s)
    
    result = pd.DataFrame(
    
        {
     'Model': model_list,
     'F_score': f_score,
     'Accuracy': acc_list,
     'Balanced Acc': bacc_list,
     'Cohen Kappa': cohen_list,
     'Log Loss': ll_list,
     'Roc_AUC_Score': roc_auc_list,
     'Confusion Matrix': cm_list,
     'Precision': pr_list,
     'Recall': rc_list,
     'Time Taken': t_list,
     #'cv_list': cv_list 
    })
    
    return result


# In[103]:


results = result_matrix(cleaned_df)


# ## Results

# In[104]:


results

