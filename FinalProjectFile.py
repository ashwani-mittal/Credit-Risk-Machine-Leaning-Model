# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:54:26 2023

@author: Abhishek
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
file_path_data = r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\train_data.csv"
file_path_labels = r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\train_labels.csv"

data = pd.read_csv(file_path_data, nrows = 500000)

labels = pd.read_csv(file_path_labels)

print(data.shape)
print(labels.shape)
print(data.head)

data['S_2'] = pd.to_datetime(data['S_2'])
#data = data.groupby('customer_ID')
data.tail(5)

last_customer_id = data['customer_ID'].iloc[-1]
print(last_customer_id)

data2 = data[data['customer_ID'] != '425d5532a25f20815166956bc7fb3a49928f4c49517e1c59f9843f13a8aad690']
print(data2.tail)

print(data2.shape)

merged_data = pd.merge(data2, labels, on='customer_ID').dropna(subset=['target'])
merged_data.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\merged-data-final.csv", index = False)

print(merged_data.iloc[:,-6:])
merged_data.iloc[:,-6:].to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\merged-data-final-Last6.csv", index = False)
data_types = merged_data.dtypes
data_types.dtype
data_types = pd.DataFrame({"label": merged_data.columns,"type": data_types})
print(data_types)

categorical_variables = data_types.loc[data_types['type']=='object']
print(categorical_variables)

categorical_variables = categorical_variables.iloc[1:]
print(categorical_variables)

cat_list = categorical_variables.iloc[:, 0].tolist()

print(cat_list)

one_hot = OneHotEncoder(handle_unknown = 'ignore')
one_hot.fit(merged_data[cat_list])
merged_data_encoded = pd.DataFrame(one_hot.transform(merged_data[cat_list]).toarray(),columns = one_hot.get_feature_names_out(cat_list))
merged_data = pd.concat([merged_data.reset_index(drop = True), merged_data_encoded.reset_index(drop=True)],axis = 1)
print(merged_data)

print(merged_data['S_2'].min())
print(merged_data['S_2'].max())

train_data = merged_data[(merged_data['S_2']>='2017-05-01') & (merged_data["S_2"] <= '2018-01-31')]
test1_data = merged_data[(merged_data['S_2']>='2017-03-01') & (merged_data['S_2'] <= '2017-04-30')]
test2_data = merged_data[(merged_data['S_2']>='2018-02-01') & (merged_data['S_2'] <= '2018-03-31')]

print(train_data['S_2'].min())
print(train_data['S_2'].max())
print(test1_data['S_2'].min())
print(test1_data['S_2'].max())
print(test2_data['S_2'].min())
print(test2_data['S_2'].max())

target_col = 'target'
drop_cols = ['customer_ID','S_2', 'D_63','D_64',target_col]
X_train = train_data.drop(columns = drop_cols)
Y_train = train_data[target_col]

xgb_model_default = xgb.XGBClassifier(random_state = 42)
xgb_model_default.fit(X_train, Y_train)

importance_df = pd.DataFrame(xgb_model_default.feature_importances_, index = X_train.columns, columns=['importance'])
importance_df = importance_df.sort_values('importance', ascending=False)

print(importance_df)
importance_df.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\feature_importance.csv")

params = {'learning_rate': 0.5,
          'max_depth': 4,
          'subsample': 0.5,
          'colsample_bytree':0.5,
          'scale_pos_weight': 5,
          'random_state': 42}

# Set the number of trees
num_trees = 300

# Build the XGBoost model
xgb_model_new = xgb.XGBClassifier(n_estimators=num_trees, **params)
xgb_model_new.fit(X_train,Y_train)

importance_df2 = pd.DataFrame(xgb_model_new.feature_importances_, index = X_train.columns, columns=['importance'])
importance_df2 = importance_df2.sort_values('importance', ascending=False)
print(importance_df2)

importance_df2.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\feature_importance2.csv")

importance_df2 = pd.read_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\feature_importance2.csv", index_col= "Unnamed: 0")

# Merge the two dataframes on the index column
merged_importance_df = importance_df.merge(importance_df2, left_index=True, right_index=True)

# Rename the importance columns
merged_importance_df = merged_importance_df.rename(columns={'importance_x': 'importance_1', 'importance_y': 'importance_2'})

# Add a column for the feature names
merged_importance_df['feature_name'] = merged_importance_df.index

# Reorder the columns to have the feature_name column on the left
merged_importance_df = merged_importance_df[['feature_name', 'importance_1', 'importance_2']]

# Reset the index
merged_importance_df = merged_importance_df.reset_index(drop=True)

print(merged_importance_df)

merged_importance_df.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\merged_importance.csv")

# Create a boolean mask for features with importance greater than 0.5% in either column
mask = (merged_importance_df['importance_1'] > 0.005) | (merged_importance_df['importance_2'] > 0.005)

# Get the feature names where the mask is True
important_features = merged_importance_df.loc[mask, 'feature_name'].tolist()
print(len(important_features))


# Filter the X_train dataframe based on the columns to keep
X_train_filtered = X_train[important_features]
print(X_train_filtered.shape)


#We now have only 41 features remaining in our filtered X_train dataframe. We will now proceed to create x_test and y_test 1 and 2 dataframes to be used in grid search

X_test1 = test1_data.drop(columns = drop_cols)
X_test1 = X_test1[important_features]
Y_test1 = test1_data[target_col]
X_test2 = test2_data.drop(columns = drop_cols)
X_test2 = X_test2[important_features]
Y_test2 = test2_data[target_col]

#test to see if shapes of train and test samples are the same
print(X_train_filtered.shape)
print(Y_train.shape)
print(X_test1.shape)
print(X_test2.shape)
print(Y_test1.shape)
print(Y_test2.shape)

from sklearn.model_selection import ParameterGrid
clf = xgb.XGBClassifier(objective='binary:logistic', seed=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 300],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.5, 1.0],
    'scale_pos_weight': [1, 5, 10]
}

# Initialize the grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Initialize the dataframe to store the results
results_table = pd.DataFrame(columns=["Trees", "LR", "Subsample", "% Features", "Weight of Default", "AUC Train", "AUC Test 1", "AUC Test 2"])

# Loop over all parameter combinations
for params in ParameterGrid(param_grid):
    # Set the parameters for the classifier
    clf.set_params(**params)
    
    # Fit the classifier to the training data
    clf.fit(X_train_filtered, Y_train)
    
    # Compute the AUC score on the training set
    y_train_pred = clf.predict_proba(X_train_filtered)[:, 1]
    auc_train = roc_auc_score(Y_train, y_train_pred)
    
    # Compute the AUC score on the first test set (This will be also validation error)
    y_test1_pred = clf.predict_proba(X_test1)[:, 1]
    auc_test1 = roc_auc_score(Y_test1, y_test1_pred)
    
    # Compute the AUC score on the second test set (This will be also validation error)
    y_test2_pred = clf.predict_proba(X_test2)[:, 1]
    auc_test2 = roc_auc_score(Y_test2, y_test2_pred)
    
    # Append the results to the dataframe
    results_table = results_table.append({
        "Trees": params['n_estimators'],
        "LR": params['learning_rate'],
        "Subsample": params['subsample'],
        "% Features": params['colsample_bytree'],
        "Weight of Default": params['scale_pos_weight'],
        "AUC Train": auc_train,
        "AUC Test 1": auc_test1,
        "AUC Test 2": auc_test2
    }, ignore_index=True)
    
    # Print the current results
    print(results_table.iloc[-1])
    
results_table.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\XGB_GridSearch_results.csv", index= False)

Import_results_table = pd.read_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\XGB_GridSearch_results.csv")

Import_results_table.sort_values(by=["AUC_Test2", "Train_Error", "Val_Error1", "Val_Error2"], ascending=[True, False, False, False])


#compare the bias and variance of each model by looking at the training error and 
#the difference between the training error and the validation error.
#The model with the lowest training error and the smallest difference between the training error and 
#validation error is likely to have the best balance of bias and variance.
#You can also look at the AUC score on the test set to determine the overall performance of the model,
#but it's important to keep in mind that AUC score alone does not provide information about bias and variance.


print(X_train_filtered.columns.duplicated())

X_train_filtered = X_train_filtered.loc[:, ~X_train_filtered.columns.duplicated()]
print(Import_results_table)
print(Import_results_table.iloc[0][0:5].to_dict())

best_model_params = Import_results_table.iloc[0][0:5].to_dict()
print(best_model_params)
best_model = xgb.XGBClassifier(objective='binary:logistic', seed=42)
best_model.set_params(**best_model_params)
best_model.fit(X_train_filtered, Y_train)


bestparams = {'learning_rate': 0.01,
          'subsample': 0.5,
          'colsample_bytree':0.5,
          'scale_pos_weight': 1.0,
          'random_state': 42}

# Set the number of trees
num_trees = 50

# Build the XGBoost model
xgb_model_best = xgb.XGBClassifier(n_estimators=num_trees, **bestparams)
xgb_model_best.fit(X_train_filtered,Y_train)

xgb_model_best.predict_proba(X_train_filtered)

# Rank Ordering
perf_train_data = pd.DataFrame({"Actual": train_data['target'], "Prediction": xgb_model_best.predict_proba(X_train_filtered)[:,1]})
quantiles = list(set(perf_train_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)

perf_train_data["Score Bins"] = pd.cut(perf_train_data["Prediction"], quantiles)
stat = perf_train_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat["Bad Rate"] = stat["sum"] / stat["count"]
stat

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(stat.index.astype(str), stat["Bad Rate"])
ax.set_xlabel("Score Bins")
ax.set_ylabel("Default Rate")
ax.set_title("Default Rate by Score Bins (Train)")
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\rank_ordering_xgb_train.jpg")


X_test1 = X_test1.loc[:, ~X_test1.columns.duplicated()]

# Rank Ordering
perf_test1_data = pd.DataFrame({"Actual": test1_data['target'], "Prediction": xgb_model_best.predict_proba(X_test1)[:,1]})
quantiles = list(set(perf_test1_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)

perf_test1_data["Score Bins"] = pd.cut(perf_test1_data["Prediction"], quantiles)
stat_test1 = perf_test1_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat_test1["Bad Rate"] = stat_test1["sum"] / stat_test1["count"]
stat_test1

fig, ax = plt.subplots()
ax.bar(stat_test1.index.astype(str), stat_test1["Bad Rate"])
ax.set_xlabel("Score Bins")
ax.set_ylabel("Default Rate")
ax.set_title("Default Rate by Score Bins (Test1)")
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\rank_ordering_xgb_test1.png")


X_test2 = X_test2.loc[:, ~X_test2.columns.duplicated()]

perf_test2_data = pd.DataFrame({"Actual": test2_data['target'], "Prediction": xgb_model_best.predict_proba(X_test2)[:,1]})
quantiles = list(set(perf_test1_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)

perf_test2_data["Score Bins"] = pd.cut(perf_test2_data["Prediction"], quantiles)
stat_test2 = perf_test2_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat_test2["Bad Rate"] = stat_test2["sum"] / stat_test2["count"]
stat_test2

fig, ax = plt.subplots()
ax.bar(stat_test2.index.astype(str), stat_test2["Bad Rate"])
ax.set_xlabel("Score Bins")
ax.set_ylabel("Default Rate")
ax.set_title("Default Rate by Score Bins (Test2)")
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\rank_ordering_xgb_test2")










import joblib

joblib.dump(best_model, "xgb_model.joblib")

from joblib import load

#Load the saved model from a file
best_model = load('best_model.joblib')

##### Now Lets do it for Neural Network #####

#Missing value imputation
merged_data.fillna(0,inplace=True)

train_data_NN = merged_data[(merged_data['S_2']>='2017-05-01') & (merged_data["S_2"] <= '2018-01-31')]
test1_data_NN = merged_data[(merged_data['S_2']>='2017-03-01') & (merged_data['S_2'] <= '2017-04-30')]
test2_data_NN = merged_data[(merged_data['S_2']>='2018-02-01') & (merged_data['S_2'] <= '2018-03-31')]

target_col = 'target'
drop_cols = ['customer_ID','S_2', 'D_63','D_64',target_col]

X_train_NN = train_data_NN.drop(columns = drop_cols)
X_train_NN = X_train_NN[important_features]
Y_train_NN = train_data_NN[target_col]

X_test1_NN = test1_data_NN.drop(columns = drop_cols)
X_test1_NN = X_test1_NN[important_features]
Y_test1_NN = test1_data_NN[target_col]

X_test2_NN = test2_data_NN.drop(columns = drop_cols)
X_test2_NN = X_test2_NN[important_features]
Y_test2_NN = test2_data_NN[target_col]

#Normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train_NN)


X_train_NN_normalized = sc.transform(X_train_NN)
X_test1_NN_normalized = sc.transform(X_test1_NN)
X_test2_NN_normalized = sc.transform(X_test2_NN)

X_train_NN_normalized = pd.DataFrame(X_train_NN_normalized, columns=X_train_NN.columns)
X_test1_NN_normalized = pd.DataFrame(X_test1_NN_normalized, columns=X_test1_NN.columns)
X_test2_NN_normalized = pd.DataFrame(X_test2_NN_normalized, columns=X_test2_NN.columns)

#Outlier Treatment for train sample
X_train_NN_normalized.describe(percentiles=[0.01, 0.99]).transpose()

#Removing Duplicate Columns
X_train_NN_normalized = X_train_NN_normalized.loc[:, ~X_train_NN_normalized.columns.duplicated()]
X_test1_NN_normalized = X_test1_NN_normalized.loc[:, ~X_test1_NN_normalized.columns.duplicated()]
X_test2_NN_normalized = X_test2_NN_normalized.loc[:, ~X_test2_NN_normalized.columns.duplicated()]

for col in X_train_NN_normalized.columns:
    # calculate the 1% and 99% quantiles for the column
    min_val = X_train_NN_normalized[col].quantile(0.01)
    max_val = X_train_NN_normalized[col].quantile(0.99)
    
    #replace any values outside of the range with the corresponding minimum or maximum value
    X_train_NN_normalized[col] = X_train_NN_normalized[col].apply(lambda x: min_val if x < min_val else max_val if x > max_val else x)


#Outlier Treatment for test sample 1
X_test1_NN_normalized.describe(percentiles=[0.01, 0.99]).transpose()

for col in X_train_NN_normalized.columns:
    # calculate the 1% and 99% quantiles for the column
    min_val = X_test1_NN_normalized[col].quantile(0.01)
    max_val = X_test1_NN_normalized[col].quantile(0.99)
    
    #replace any values outside of the range with the corresponding minimum or maximum value
    X_test1_NN_normalized[col] = X_test1_NN_normalized[col].apply(lambda x: min_val if x < min_val else max_val if x > max_val else x)

#Outlier Treatment for test sample 2
X_test2_NN_normalized.describe(percentiles=[0.01, 0.99]).transpose()

for col in X_train_NN_normalized.columns:
    # calculate the 1% and 99% quantiles for the column
    min_val = X_test2_NN_normalized[col].quantile(0.01)
    max_val = X_test2_NN_normalized[col].quantile(0.99)
    
    #replace any values outside of the range with the corresponding minimum or maximum value
    X_test2_NN_normalized[col] = X_test2_NN_normalized[col].apply(lambda x: min_val if x < min_val else max_val if x > max_val else x)

import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense


# set hyperparameters
num_hidden_layers = [2, 4]
num_nodes = [4, 6]
activation_funcs = ['relu', 'tanh']
dropout_rates = [0.5, 1.0]
batch_sizes = [100, 10000]
num_epochs = 20

# create empty dataframe to store results
results_df_NN = pd.DataFrame(columns=['# HL', '#Node', 'Activation Function', 'Dropout', 'Batch Size', 'AUC Train', 'AUC Test 1', 'AUC Test 2'])

# iterate through all combinations of hyperparameters
for hl in num_hidden_layers:
    for nn in num_nodes:
        for af in activation_funcs:
            for dr in dropout_rates:
                for bs in batch_sizes:
                    
                    # build neural network model
                    model = Sequential()
                    model.add(Dense(nn, activation=af, input_shape=(X_train_NN_normalized.shape[1],)))
                    for i in range(hl - 1):
                        model.add(Dense(nn, activation=af))
                        if dr < 1.0:
                            model.add(Dropout(dr))
                    model.add(Dense(1, activation='sigmoid'))
                    model.compile(loss='binary_crossentropy', optimizer='adam')
                    
                    # train model
                    model.fit(X_train_NN_normalized, Y_train, batch_size=bs, epochs=num_epochs, verbose=0)
                    
                    # evaluate performance on train and test sets
                    y_pred_train = model.predict(X_train_NN_normalized).ravel()
                    y_pred_test1 = model.predict(X_test1_NN_normalized).ravel()
                    y_pred_test2 = model.predict(X_test2_NN_normalized).ravel()
                    auc_train = roc_auc_score(Y_train, y_pred_train)
                    auc_test1 = roc_auc_score(Y_test1, y_pred_test1)
                    auc_test2 = roc_auc_score(Y_test2, y_pred_test2)
                    
                    # add results to dataframe
                    results_df_NN = results_df_NN.append({'# HL': hl, '#Node': nn, 'Activation Function': af, 'Dropout': dr, 'Batch Size': bs, 'AUC Train': auc_train, 'AUC Test 1': auc_test1, 'AUC Test 2': auc_test2}, ignore_index=True)
                    print(results_df_NN.iloc[-1])
                    # save results to csv after each iteration
                    #results_df.to_csv('results.csv', index=False)
            

results_df_NN.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\NN_grid_results.csv")


Import_results_table_NN = pd.read_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\NN_grid_results.csv")

Import_results_table_NN.sort_values(by=["AUC_Test2", "Train_Error", "Val_Error1", "Val_Error2"], ascending=[True, False, False, False])




#set best hyperparameters
hl = 2
nn = 4
af = 'relu'
dr = 0.5
bs = 100

#fit the 
model = Sequential()
model.add(Dense(nn, activation=af, input_shape=(X_train_NN_normalized.shape[1],)))
for i in range(hl - 1):
    model.add(Dense(nn, activation=af))
    if dr < 1.0:
        model.add(Dropout(dr))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')              
model.fit(X_train_NN_normalized, Y_train, batch_size=bs, epochs=20, verbose=0)

model.predict(X_train_NN_normalized)


# Rank Ordering
perf_train_data_NN = pd.DataFrame({"Actual": train_data['target'], "Prediction": model.predict(X_train_NN_normalized)[:,0]})
quantiles_NN = list(set(perf_train_data_NN.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles_NN.sort()
quantiles_NN.insert(0,0)
quantiles_NN.insert(len(quantiles),1)

perf_train_data_NN["Score Bins"] = pd.cut(perf_train_data_NN["Prediction"], quantiles)
stat_NN = perf_train_data_NN.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat_NN["Bad Rate"] = stat_NN["sum"] / stat_NN["count"]
stat_NN

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(stat_NN.index.astype(str), stat["Bad Rate"])
ax.set_xlabel("Score Bins")
ax.set_ylabel("Default Rate")
ax.set_title("Default Rate by Score Bins (Train)")
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\rank_ordering_NN_train.jpg")


#X_test1 = X_test1.loc[:, ~X_test1.columns.duplicated()]

# Rank Ordering
perf_test1_data= pd.DataFrame({"Actual": test1_data['target'], "Prediction": model.predict(X_test1_NN_normalized)[:,0]})
quantiles = list(set(perf_test1_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)

perf_test1_data["Score Bins"] = pd.cut(perf_test1_data["Prediction"], quantiles)
stat_test1 = perf_test1_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat_test1["Bad Rate"] = stat_test1["sum"] / stat_test1["count"]
stat_test1

fig, ax = plt.subplots()
ax.bar(stat_test1.index.astype(str), stat_test1["Bad Rate"])
ax.set_xlabel("Score Bins")
ax.set_ylabel("Default Rate")
ax.set_title("Default Rate by Score Bins (Test1)")
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\rank_ordering_NN_test1.png")


#X_test2 = X_test2.loc[:, ~X_test2.columns.duplicated()]

perf_test2_data = pd.DataFrame({"Actual": test2_data['target'], "Prediction": model.predict(X_test2_NN_normalized)[:,0]})
quantiles = list(set(perf_test1_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)

perf_test2_data["Score Bins"] = pd.cut(perf_test2_data["Prediction"], quantiles)
stat_test2 = perf_test2_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat_test2["Bad Rate"] = stat_test2["sum"] / stat_test2["count"]
stat_test2

fig, ax = plt.subplots()
ax.bar(stat_test2.index.astype(str), stat_test2["Bad Rate"])
ax.set_xlabel("Score Bins")
ax.set_ylabel("Default Rate")
ax.set_title("Default Rate by Score Bins (Test2)")
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\rank_ordering_NN_test2")






















































#Strategy; Applicants with probability of default (model’s output) lower than threshold, will be accepted,
#and those with PD higher than threshold will be rejected. 
#Aggresive Strategy; Higher threshold, Hence accept more applicants.
#Conservative Strategy; Lower Threshold, Hence accept less applicants.
#For strategy purposes we will use X_Train_Filtered because we have already used Test_1 and Test_2 data now
#we do not want our model to be over fitted to the tests sets and we want to make sure we are not biasing our model
#based on the performance of the test sets.









#Function to Calculate Default rate and Revenue
def calculate_metrics(data, target_var, threshold, spend_feature, balance_feature):
    # calculate default rate
    accepted = data[data[target_var] < threshold]
    default_rate = len(accepted[accepted['default_payment_next_month']==1])/len(accepted)
    
    # calculate revenue
    accepted_no_default = accepted[accepted['default_payment_next_month']==0]
    revenue = (accepted_no_default[spend_feature].sum()*0.001)+(accepted_no_default[balance_feature].sum()*0.02)
    
    return default_rate, revenue

# assuming the target variable is called "default_payment_next_month"
# and we chose "S_LIMIT_BAL" as the spend feature and "B_PAY_AMT1" as the balance feature
train_default_rate, train_revenue = calculate_metrics(train_data, "default_payment_next_month", 0.5, "S_LIMIT_BAL", "B_PAY_AMT1")


# Filter the merged_data dataframe based on the columns to keep
merged_data_filtered = merged_data[important_features]

#Removing Duplicates
merged_data_filtered = merged_data_filtered.loc[:, ~merged_data_filtered.columns.duplicated()]


import shap

# initialize the explainer with the training data
explainer = shap.TreeExplainer(best_model)

# calculate SHAP values for the test data
shap_values = explainer.shap_values(merged_data_filtered)

# calculate the mean SHAP values for all features
shap_mean = np.abs(shap_values).mean(axis=0)

# find the indices of the top 5 features
top_features = np.argsort(shap_mean)[-5:]

# get the summary statistics for each of the top 5 features
summary_stats = []
for i in top_features:
    feature_name = merged_data_filtered.columns[i]
    feature_values = merged_data_filtered[feature_name]
    feature_summary_stats = feature_values.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    percent_missing = (1 - feature_values.count() / len(feature_values)) * 100
    feature_summary_stats['% Missing'] = percent_missing
    feature_summary_stats = feature_summary_stats.rename(feature_name)
    summary_stats.append(feature_summary_stats)

# combine the summary statistics into a single DataFrame
summary_stats_df = pd.concat(summary_stats, axis=1)

#Result came in rows not columns so transposing it
summary_stats_df = summary_stats_df.T


# reorder the columns
summary_stats_df = summary_stats_df[['min', '1%', '5%', '50%', '95%', '99%', 'max', 'mean', '% Missing']]

# rename the columns
summary_stats_df.columns = ['Min', '1 Percentile', '5 Percentile', 'Median', '95 Percentile', '99 Percentile', 'Max', 'Mean', '% Missing']

# add a column for the feature names
summary_stats_df['Feature'] = summary_stats_df.index

# reorder the columns
summary_stats_df = summary_stats_df[['Feature', 'Min', '1 Percentile', '5 Percentile', 'Median', '95 Percentile', '99 Percentile', 'Max', 'Mean', '% Missing']]

# Store the file to csv
summary_stats_df.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\shap_summary_stats.csv", index= False)



#importance = best_model.feature_importances_
#feature_names = X_test2.columns.tolist()[-5:]

#df = pd.DataFrame({'feature': feature_names, 'importance': importance})


import seaborn as sns

#sns.set(style="whitegrid")
#sns.swarmplot(x="importance", y="feature", data=df)


sns.set(style="whitegrid")

sns.swarmplot(x="variable", y="value", data=pd.melt(X_test2[top_features]), hue="output", palette="Set2")



import shap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# assume that you have already trained an XGBoost model
# and loaded the test data into a pandas DataFrame called `df_test2`
# randomly sample a subset of the test data


bswarm_test2 = X_test2_NN.drop(columns = drop_cols)
test_data_subset = X_test2.sample(n=1000, random_state=42)

# use the SHAP library to compute SHAP values for each feature on the test data subset
explainer = shap.Explainer(model)
shap_values = explainer(test_data_subset)


# create a DataFrame of the SHAP values and feature names
shap_df = pd.DataFrame(shap_values.values, columns=test_data_subset.columns)


# calculate the absolute SHAP values for each feature
abs_shap_df = shap_df.abs()


# calculate the mean absolute SHAP value for each feature
mean_abs_shap_df = abs_shap_df.mean()


# sort the features by their mean absolute SHAP value, in descending order
sorted_idx = mean_abs_shap_df.argsort()[::-1]


# select the top 10 features with the highest mean absolute SHAP value
top_n = 5
top_features = test_data_subset.columns[sorted_idx][:top_n]

# create the beeswarm plot using the selected features
sns.set(style="whitegrid")
plt.figure(figsize=(12,8))
sns.swarmplot(data=shap_df[top_features], orient='h')
plt.xlabel("SHAP value")
plt.title(f"Top {top_n} Features based on Mean Absolute SHAP Value")
plt.show()






#####ForXGBoost WaterFall###
import plotly.graph_objs as go

important_features_WF = X_test2[top_features]

print(important_features_WF)
feature_values = important_features_WF.values

print(feature_values)
# Calculate the y-values

#y_values = important_features_WF.cumsum(axis=1)[:, ::5]

sliced_df = important_features_WF.iloc[:, ::5]
#print(y_values)
#y_values.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\y_values.csv", index= False)


# Create trace for the bars
#trace = go.Waterfall(name='Waterfall', orientation='v',
                     #x=shap_df[top_features], y=important_features_WF,
                     #decreasing={'marker': {'color': 'red'}},
                     #increasing={'marker': {'color': 'green'}},
                     #totals={'marker': {'color': 'blue', 'line': {'color': 'blue', 'width': 3}}},
                     #textposition='outside')

# Create layout for the chart
#layout = go.Layout(title='Waterfall Chart', xaxis={'title': 'Item'}, yaxis={'title': 'Value'})

# Create figure and plot
#fig = go.Figure(data=[trace], layout=layout)
#fig.show()

# create a sample DataFrame with column names and values
#data = pd.DataFrame({
    #'Category': ['Initial Value', 'Category A', 'Category B', 'Category C', 'Category D', 'Final Value'],
    #'Value': [1000, 500, -200, -300, 600, None]
#})


from waterfall_chart import plot

#First get one observation from the data
important_features_WF_1 = important_features_WF.iloc[1:2]
drop_important_features_wf_1 = important_features_WF_1.drop('Value', axis=1)
print(drop_important_features_wf_1.columns)
print(important_features_WF_1)

drop_important_features_wf_1 =drop_important_features_wf_1.T

feature_values_1 = drop_important_features_wf_1.values
print(feature_values_1)

# set the first and last values to 0 to start and end the waterfall chart at 0
#important_features_WF_1.loc[0, 'Value'] = 0
#important_features_WF_1.loc[len(important_features_WF_1) - 1, 'Value'] = 0

# set the index to the category column for plotting
drop_important_features_wf_1.set_index(drop_important_features_wf_1.index, inplace=True)


# plot the waterfall chart
plot(drop_important_features_wf_1.index, feature_values_1.flatten(), rotation_value=45)


#First Chose the Balance feature and Spend Feature
balance_feature = 'B_9'
spend_feature = 'S_3'

#Calculate probability of default for all observations in the train sample
pd_train = model.predict(X_train_NN_normalized)[:,0]
pd_train.min()

#create a dataframe for the train sample with probability of default and the actual default
train_results = X_train_NN_normalized.copy()
train_results = train_results.loc[:,[balance_feature,spend_feature]]
train_results['Probability of Default'] = pd_train
train_results['Defaulted'] = Y_train
print(len(train_results['Defaulted']))
print(train_results['Defaulted'].sum())
train_results = train_results.reset_index(drop=True)
train_results


# Define a list of thresholds to iterate over
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Create an empty dataframe to store the results
train_sample_threshold_results = pd.DataFrame(columns=['threshold', '#total observations', '# of observations where got_loan = Yes', '# of observations where got_loan = Yes and customer default ==1', 'revenue'])

def calculate_revenue(row):
    if row['Did the customer get a loan'] == 'Yes':
        if row['Defaulted'] == 1:
            return 0
        else:
            return row['B_9'] * 0.02 + row['S_3'] * 0.001
    else:
        return 0

# Iterate over the list of thresholds
for threshold in thresholds:

    # Create a new dataframe with required columns
    new_df = pd.DataFrame(columns=['B_9', 'S_3', 'Probability of Default', 'Defaulted', 'Did the customer get a loan', 'Revenue', 'Threshold'])

    new_df[['B_9', 'S_3', 'Probability of Default', 'Defaulted']] = train_results[['B_9', 'S_3', 'Probability of Default', 'Defaulted']]
    new_df['Did the customer get a loan'] = new_df['Probability of Default'].apply(lambda x: 'Yes' if x < threshold else 'No')
    new_df['Revenue'] = new_df.apply(calculate_revenue, axis=1)
    new_df['Threshold'] = threshold

    # Group the new dataframe by the threshold and calculate the required statistics
    grouped_df = new_df.groupby(['Threshold']).agg({'B_9': 'count', 'Did the customer get a loan': lambda x: x[x == 'Yes'].count(), 'Defaulted': lambda x: ((x < threshold) & (x == 1)).sum(), 'Revenue': 'sum'})

    # Add a new row to the results dataframe
    train_sample_threshold_results = pd.concat([train_sample_threshold_results, pd.DataFrame({'threshold': threshold, '#total observations': grouped_df['B_9'].sum(), '# of observations where got_loan = Yes': grouped_df['Did the customer get a loan'].sum(), '# of observations where got_loan = Yes and customer default ==1': ((new_df['Defaulted'] == 1) & (new_df['Did the customer get a loan'] == 'Yes')).sum(), 'revenue': grouped_df['Revenue'].sum()}, index=[0])], ignore_index=True)

# Print the results dataframe
train_sample_threshold_results

train_sample_threshold_results.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\train_sample_threshold.csv")


#calculate probability of default for all observations in the test1 sample
pd_test1 = model.predict(X_test1_NN_normalized)[:,0]
pd_test1

#create a dataframe for the train sample with probability of default and the actual default
test1_results = X_test1_NN_normalized.copy()
test1_results = test1_results.loc[:,[balance_feature,spend_feature]]
test1_results['Probability of Default'] = pd_test1
test1_results['Defaulted'] = Y_test1
print(len(test1_results['Defaulted']))
print(test1_results['Defaulted'].sum())
test1_results

# Define a list of thresholds to iterate over
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Create an empty dataframe to store the results
test1_sample_threshold_results = pd.DataFrame(columns=['threshold', '#total observations', '# of observations where got_loan = Yes', '# of observations where got_loan = Yes and customer default ==1', 'revenue'])

def calculate_revenue(row):
    if row['Did the customer get a loan'] == 'Yes':
        if row['Defaulted'] == 1:
            return 0
        else:
            return row['B_9'] * 0.02 + row['S_3'] * 0.001
    else:
        return 0

# Iterate over the list of thresholds
for threshold in thresholds:

    # Create a new dataframe with required columns
    new_df = pd.DataFrame(columns=['B_9', 'S_3', 'Probability of Default', 'Defaulted', 'Did the customer get a loan', 'Revenue', 'Threshold'])

    new_df[['B_9', 'S_3', 'Probability of Default', 'Defaulted']] = test1_results[['B_9', 'S_3', 'Probability of Default', 'Defaulted']]
    new_df['Did the customer get a loan'] = new_df['Probability of Default'].apply(lambda x: 'Yes' if x < threshold else 'No')
    new_df['Revenue'] = new_df.apply(calculate_revenue, axis=1)
    new_df['Threshold'] = threshold

    # Group the new dataframe by the threshold and calculate the required statistics
    grouped_df = new_df.groupby(['Threshold']).agg({'B_9': 'count', 'Did the customer get a loan': lambda x: x[x == 'Yes'].count(), 'Defaulted': lambda x: ((x < threshold) & (x == 1)).sum(), 'Revenue': 'sum'})

    # Add a new row to the results dataframe
    test1_sample_threshold_results = pd.concat([test1_sample_threshold_results, pd.DataFrame({'threshold': threshold, '#total observations': grouped_df['B_9'].sum(), '# of observations where got_loan = Yes': grouped_df['Did the customer get a loan'].sum(), '# of observations where got_loan = Yes and customer default ==1': ((new_df['Defaulted'] == 1) & (new_df['Did the customer get a loan'] == 'Yes')).sum(), 'revenue': grouped_df['Revenue'].sum()}, index=[0])], ignore_index=True)

# Print the results dataframe
test1_sample_threshold_results

test1_sample_threshold_results.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\test1_sample_threshold.csv")

pd_test2 = model.predict(X_test2_NN_normalized)[:,0]
pd_test2


#create a dataframe for the train sample with probability of default and the actual default
test2_results = X_test2_NN_normalized.copy()
test2_results = test2_results.loc[:,[balance_feature,spend_feature]]
test2_results['Probability of Default'] = pd_test2
test2_results['Defaulted'] = Y_test2
print(len(test2_results['Defaulted']))
print(test2_results['Defaulted'].sum())
test2_results


# Define a list of thresholds to iterate over
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Create an empty dataframe to store the results
test2_sample_threshold_results = pd.DataFrame(columns=['threshold', '#total observations', '# of observations where got_loan = Yes', '# of observations where got_loan = Yes and customer default ==1', 'revenue'])

def calculate_revenue(row):
    if row['Did the customer get a loan'] == 'Yes':
        if row['Defaulted'] == 1:
            return 0
        else:
            return row['B_9'] * 0.02 + row['S_3'] * 0.001
    else:
        return 0

# Iterate over the list of thresholds
for threshold in thresholds:

    # Create a new dataframe with required columns
    new_df = pd.DataFrame(columns=['B_9', 'S_3', 'Probability of Default', 'Defaulted', 'Did the customer get a loan', 'Revenue', 'Threshold'])

    new_df[['B_9', 'S_3', 'Probability of Default', 'Defaulted']] = test2_results[['B_9', 'S_3', 'Probability of Default', 'Defaulted']]
    new_df['Did the customer get a loan'] = new_df['Probability of Default'].apply(lambda x: 'Yes' if x < threshold else 'No')
    new_df['Revenue'] = new_df.apply(calculate_revenue, axis=1)
    new_df['Threshold'] = threshold

    # Group the new dataframe by the threshold and calculate the required statistics
    grouped_df = new_df.groupby(['Threshold']).agg({'B_9': 'count', 'Did the customer get a loan': lambda x: x[x == 'Yes'].count(), 'Defaulted': lambda x: ((x < threshold) & (x == 1)).sum(), 'Revenue': 'sum'})

    # Add a new row to the results dataframe
    test2_sample_threshold_results = pd.concat([test2_sample_threshold_results, pd.DataFrame({'threshold': threshold, '#total observations': grouped_df['B_9'].sum(), '# of observations where got_loan = Yes': grouped_df['Did the customer get a loan'].sum(), '# of observations where got_loan = Yes and customer default ==1': ((new_df['Defaulted'] == 1) & (new_df['Did the customer get a loan'] == 'Yes')).sum(), 'revenue': grouped_df['Revenue'].sum()}, index=[0])], ignore_index=True)

# Print the results dataframe
test2_sample_threshold_results


test2_sample_threshold_results.to_csv(r"C:\Users\ali\OneDrive\Desktop\Education\Machine Learning\Project Files\test2_sample_threshold.csv")















