#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[3]:


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced


# In[4]:


file_path = Path("C:/Users/fizza/Documents/LoanStats_2019Q1_11.csv")
df = pd.read_csv(file_path)


df.head()


# In[5]:


# Convert string values to binary
df_encoded = pd.get_dummies(df, columns=["home_ownership", "verification_status", "issue_d", "pymnt_plan", "initial_list_status", "next_pymnt_d", "application_type", "hardship_flag", "debt_settlement_flag"], drop_first=True)
df_encoded.head()


# In[6]:


# Saving the encoded dataset
# csv_path = Path("C:/Users/fizza/LoanStats_2019Q1_1.csv")
# df_encoded.to_csv(csv_path, index=False)


# In[7]:


# Create our features
X = df_encoded.copy()
X = X.drop(columns=["loan_status"],axis=1)
columns_names = X.dtypes[X.dtypes=='object'].index
X = pd.get_dummies(X,columns=columns_names)


# Create our target
y = df_encoded["loan_status"]


# In[8]:


X.describe()


# In[9]:


# Check the balance of our target values
y.value_counts()


# In[10]:


# Split the X and y into X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X.shape


# In[11]:


# Create the StandardScaler instance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[12]:


# Fit the Standard Scaler with the training data
# When fitting scaling functions, only train on the training dataset
X_scaler =scaler.fit(X_train)
X_scaler


# In[13]:


# Scale the training and testing data
# X_train = X_scaler.transform(X_train)
# X_test = X_scaler.transform(X_test)


# In[14]:


# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
X_train_scaled.shape


# In[15]:


# Confirm scaled values...for fun
df_X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
round(df_X_train_scaled.mean(), 0)


# In[16]:


round(df_X_train_scaled.std(), 0)


# In[17]:


# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier


# In[18]:


rf = RandomForestClassifier()
# train the model
rf.fit(X_train_scaled, y_train)
# get predictions
predictions = rf.predict(X_test)


# In[19]:


# Resample the training data with the RandomForestClassifier
brf = RandomForestClassifier(n_estimators=100, random_state=1)
brf.fit(X_train_scaled, y_train)


# In[20]:


# Calculated the balanced accuracy score
y_pred_brf = brf.predict(X_test_scaled)
bas_brf=balanced_accuracy_score(y_test, y_pred_brf)
print(bas_brf)


# In[21]:


# Display the confusion matrix
cm_brf = confusion_matrix(y_test, y_pred_brf)
cm_df_brf = pd.DataFrame(
    cm_brf, index=["Actual High Risk", "Actual Low Risk"], columns=["Predicted High Risk", "Predicted Low Risk"]
)
cm_df_brf


# In[22]:


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred_brf))


# In[23]:


# List the features sorted in descending order by feature importance
importances = brf.feature_importances_
sorted(zip(brf.feature_importances_, X.columns), reverse=True)


# In[ ]:


# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(n_estimators=1000, random_state=1)
eec.fit(X_train_scaled, y_train)


# In[ ]:


# Calculated the balanced accuracy score
y_pred_eec = eec.predict(X_test_scaled)
bas_eec=balanced_accuracy_score(y_test, y_pred_brf)
print(bas_eec)


# In[ ]:


# Display the confusion matrix
cm_eec = confusion_matrix(y_test, y_pred_eec)
cm_df_eec = pd.DataFrame(
    cm_eec, index=["Actual High Risk", "Actual Low Risk"], columns=["Predicted High Risk", "Predicted Low Risk"]
)
cm_df_eec

