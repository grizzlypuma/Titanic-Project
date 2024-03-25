#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error



# In[52]:


data = pd.read_csv("C:/Users/kazja/OneDrive/Documents/Machine Learning Specalization/titanic/train.csv")
data


# In[53]:


data.isna().sum()


# In[54]:


df = data.dropna(subset=['Embarked'])
df.isna().sum()


# In[55]:


cat_features = ["Pclass", "Sex", "Embarked"]
df = pd.get_dummies(data = df, prefix = cat_features, columns = cat_features, dtype = int)


# In[56]:


sns.histplot(df['Age'], kde = True, bins = 30, color="Red")
count_NaN_ages = df["Age"].isna().sum()

print(f"The number of NaN values in Ages feature is {count_NaN_ages}. This is a {(count_NaN_ages / len(df['Age'])) * 100:.2f}% of all test dataset")


# In[57]:


plt.figure(figsize = (8,4))
sns.scatterplot(data = df, x = df["Age"], y = df["Fare"], alpha = 0.5)
plt.title('Scatter Plot of Age vs Fare')
plt.show()


# In[58]:


df['Pclass'] = df[['Pclass_1', 'Pclass_2', 'Pclass_3']].idxmax(axis=1)

# Now, plot the data with Seaborn
plt.figure(figsize = (8,4))
sns.scatterplot(data=df, x='Age', y='Pclass', hue='Pclass', style='Pclass', s=100, palette='deep')
plt.title('Age vs Pclass')
plt.xlabel('Age')
plt.ylabel('Pclass')
plt.grid(True)
plt.show()


# In[59]:


df['Embarked'] = df[['Embarked_C', 'Embarked_Q', 'Embarked_S']].idxmax(axis=1)
plt.figure(figsize = (8,4))
sns.scatterplot(data = df, x = "Age", y = "Embarked", hue = "Embarked", style = "Embarked")
plt.title('Age vs Embarked')


# In[60]:


plt.figure(figsize = (8,4))
sns.scatterplot(data = df, x="Age", y = "SibSp", hue = "SibSp", style = "SibSp")
plt.title('Age vs SibSp')


# In[61]:


plt.figure(figsize = (8,4))
sns.scatterplot(data = df, x="Age", y = "Parch", hue = "Parch", style = "Parch")
plt.title('Age vs Parch')


# In[62]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Fare"]].values)

df["Fare_norm"] = X_scaled


# In[76]:


features_train = ["SibSp", "Parch", "Fare_norm", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_male", "Sex_female", "Embarked_C", "Embarked_Q", "Embarked_S"]

X_train_age_full = df[df["Age"].notna()][features_train]
y_train_age_full = df[df["Age"].notna()]["Age"]

print(f"The size of X train data is {X_train_age_full.shape} and the size of y train dataset is {y_train_age_full.shape}")

print(f"The X train dataset contains {X_train_age_full.isna().sum().sum()} Nan values. y trains data set contains {y_train_age_full.isna().sum()} NaN values")


# In[77]:


from sklearn.model_selection import train_test_split

X_train_age, X_val_age, y_train_age, y_val_age = train_test_split(X_train_age_full, y_train_age_full, train_size = 0.8)

print(f"Shape of X train: {X_train_age.shape}, shape of X val is {X_val_age.shape}, shape of y_train is {y_train_age.shape}, the shape of y val is {y_val_age.shape}")


# In[65]:


min_samples_split_list = [2,5, 15, 25, 50, 100, 150, 350]                                           
max_depth_list = [1, 2, 4, 8, 16, 32, None]
n_estimators_list = [5,25,50,250]


# In[66]:


mse_list_train_age = []
mse_list_val_age =[]

for min_samples_split in min_samples_split_list:
    model = RandomForestRegressor(min_samples_split = min_samples_split).fit(X_train_age, y_train_age)
    prediction_train_age = model.predict(X_train_age)
    mse_train_age = mean_squared_error(y_train_age, prediction_train_age)
    mse_list_train_age.append(mse_train_age)
    prediction_val_age = model.predict(X_val_age)
    mse_val_age = mean_squared_error(y_val_age, prediction_val_age)
    mse_list_val_age.append(mse_val_age)

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('MSE Age')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
plt.plot(mse_list_train_age)
plt.plot(mse_list_val_age)
plt.legend(['Train','Validation'])


# In[67]:


mse_list_train_age = []
mse_list_val_age =[]

for max_depth in max_depth_list:
    model = RandomForestRegressor(max_depth=max_depth).fit(X_train_age, y_train_age)
    prediction_train_age = model.predict(X_train_age)
    mse_train_age = mean_squared_error(y_train_age, prediction_train_age)
    mse_list_train_age.append(mse_train_age)
    prediction_val_age = model.predict(X_val_age)
    mse_val_age = mean_squared_error(y_val_age, prediction_val_age)
    mse_list_val_age.append(mse_val_age)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('MSE Age')
plt.xticks(ticks = range(len(max_depth_list)),labels=max_depth_list) 
plt.plot(mse_list_train_age)
plt.plot(mse_list_val_age)
plt.legend(['Train','Validation'])


# In[68]:


mse_list_train_age = []
mse_list_val_age =[]

for n_estimator in n_estimators_list:
    model = RandomForestRegressor(n_estimators=n_estimator).fit(X_train_age, y_train_age)
    prediction_train_age = model.predict(X_train_age)
    mse_train_age = mean_squared_error(y_train_age, prediction_train_age)
    mse_list_train_age.append(mse_train_age)
    prediction_val_age = model.predict(X_val_age)
    mse_val_age = mean_squared_error(y_val_age, prediction_val_age)
    mse_list_val_age.append(mse_val_age)

plt.title('Train x Validation metrics')
plt.xlabel('N Estimator')
plt.ylabel('MSE Age')
plt.xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list) 
plt.plot(mse_list_train_age)
plt.plot(mse_list_val_age)
plt.legend(['Train','Validation'])


# In[69]:


results_RandForrest = []

for n_estimator in n_estimators_list:
    for min_samples_split in min_samples_split_list:
        for max_depth in max_depth_list:
            model = RandomForestRegressor(n_estimators = n_estimator,
                                             max_depth = max_depth, 
                                             min_samples_split = min_samples_split)
            model.fit(X_train_age, y_train_age)
            predictions_train_age = model.predict(X_train_age)
            mse_train_age = mean_squared_error(y_train_age, predictions_train_age)
            
            prediction_val_age = model.predict(X_val_age)
            mse_val_age = mean_squared_error(y_val_age, prediction_val_age)
            
            
            result_RF = {
                'n_estimator' : n_estimator,
                ' min_samples_split' :  min_samples_split,
                'max_depth' : max_depth,
                'mse_train' : mse_train_age,
                'mse_val' : mse_val_age
                
            }
            
            results_RandForrest.append(result_RF)
results_df_RF = pd.DataFrame(results_RandForrest)
results_df_RF


# In[70]:


plt.title('Train x Validation metrics')
plt.xlabel('iteration')
plt.ylabel('MSE Age')
#plt.xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list) 
plt.plot(results_df_RF['mse_train'])
plt.plot(results_df_RF['mse_val'])
plt.legend(['Train','Validation'])


# In[71]:


best_params_train_RF = results_df_RF.loc[results_df_RF['mse_train'].idxmin()]

print("Best parameters for minimum train MSE:")
print(best_params_train_RF)

best_params_val_RF = results_df_RF.loc[results_df_RF['mse_val'].idxmin()]

print("Best parameters for minimum validation MSE:")
print(best_params_val_RF)


# In[74]:


final_model_RF = RandomForestRegressor(n_estimators = 250,
                                             max_depth = 32, 
                                             min_samples_split = 2)
final_model_RF.fit(X_train_age, y_train_age)
predictions_train_age = final_model_RF.predict(X_train_age)

plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size
plt.scatter(y_train_age, predictions_train_age, alpha=0.5)  # Plots actual vs. predicted values
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Plot a diagonal line for reference
min_val = min(y_train_age.min(), predictions_train_age.min())
max_val = max(y_train_age.max(), predictions_train_age.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)  # 'k--' defines the style (black dashed line), lw is the line width

plt.show()


# ### XGBoost
# 

# In[75]:


n_estimators_list = [50,100,200,400,500,570,600]
learning_rate_list = [0.01, 0.03, 0.1, 0.3, 0.9, 1.0, 1.5]
max_depth_list = [2,4,6,7,9,10]


# In[78]:


results_XGB_fullset = []

for n_estimator in n_estimators_list:
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            model = XGBRegressor(n_estimators = n_estimator, learning_rate = learning_rate, max_depth = max_depth)
            model.fit(X_train_age_full, y_train_age_full)
            predictions_train_age = model.predict(X_train_age_full)
            mse_train = mean_squared_error(y_train_age_full, predictions_train_age)
            
            result_XGB_full = {
                'n_estimator' : n_estimator,
                'learning_rate' : learning_rate,
                'max_depth' : max_depth,
                'mse_train' : mse_train,
               
                
            }
            
            results_XGB_fullset.append(result_XGB_full)
results_df = pd.DataFrame(results_XGB_fullset)


# In[79]:


best_params_train_XGB_fullset = results_df.loc[results_df['mse_train'].idxmin()]

print("Best parameters for minimum train MSE:")
print(best_params_train_XGB_fullset)



# In[80]:


model = XGBRegressor(n_estimators = 50, learning_rate = 1.5, max_depth = 10)
model.fit(X_train_age_full, y_train_age_full)


predictions_train_age = model.predict(X_train_age_full)
mse_train = mean_squared_error(y_train_age_full, predictions_train_age)
print(f"The MSE of this model is {mse_train:.3f}")


# Assuming y_train and predictions_train are defined as your actual and predicted values
plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size
plt.scatter(y_train_age_full, predictions_train_age, alpha=0.5)  # Plots actual vs. predicted values
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Plot a diagonal line for reference
min_val = min(y_train_age_full.min(), predictions_train_age.min())
max_val = max(y_train_age_full.max(), predictions_train_age.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)  # 'k--' defines the style (black dashed line), lw is the line width

plt.show()


# In[82]:


mse_list_train = []
mse_list_val =[]

for n_estimator in n_estimators_list:
    
    model = XGBRegressor(n_estimators = n_estimator )
    model.fit(X_train_age, y_train_age)
    predictions_train_age = model.predict(X_train_age)
    mse_train = mean_squared_error(y_train_age, predictions_train_age)
    mse_list_train.append(mse_train)
    prediction_val_age = model.predict(X_val_age)
    mse_val = mean_squared_error(y_val_age, prediction_val_age)
    mse_list_val.append(mse_val)
    
plt.title('Train x Validation metrics')
plt.xlabel('N_estimators')
plt.ylabel('MSE')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list) 
plt.plot(mse_list_train)
plt.plot(mse_list_val)
plt.legend(['Train','Validation'])


# In[83]:


mse_list_train = []
mse_list_val =[]

for learning_rate in learning_rate_list:
    
    model = XGBRegressor(learning_rate = learning_rate)
    model.fit(X_train_age, y_train_age)
    predictions_train_age = model.predict(X_train_age)
    mse_train = mean_squared_error(y_train_age, predictions_train_age)
    mse_list_train.append(mse_train)
    prediction_val_age = model.predict(X_val_age)
    mse_val = mean_squared_error(y_val_age, prediction_val_age)
    mse_list_val.append(mse_val)
    
plt.title('Train x Validation metrics')
plt.xlabel('Learning_rate')
plt.ylabel('MSE')
plt.xticks(ticks = range(len(learning_rate_list )),labels=learning_rate_list) 
plt.plot(mse_list_train)
plt.plot(mse_list_val)
plt.legend(['Train','Validation'])


# In[86]:


mse_list_train = []
mse_list_val =[]

for max_depth in max_depth_list:
    
    model = XGBRegressor(max_depth = max_depth)
    model.fit(X_train_age, y_train_age)
    predictions_train_age = model.predict(X_train_age)
    mse_train = mean_squared_error(y_train_age, predictions_train_age)
    mse_list_train.append(mse_train)
    prediction_val_age = model.predict(X_val_age)
    mse_val = mean_squared_error(y_val_age,prediction_val_age)
    mse_list_val.append(mse_val)
    
plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.xticks(ticks = range(len(max_depth_list)),labels=max_depth_list) 
plt.plot(mse_list_train)
plt.plot(mse_list_val)
plt.legend(['Train','Validation'])


# In[87]:


results_XGB_splitset = []

for n_estimator in n_estimators_list:
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            model = XGBRegressor(n_estimators = n_estimator, learning_rate = learning_rate, max_depth = max_depth)
            model.fit(X_train_age, y_train_age)
            predictions_train_age = model.predict(X_train_age)
            mse_train = mean_squared_error(y_train_age, predictions_train_age)
            
            prediction_val_age = model.predict(X_val_age)
            mse_val = mean_squared_error(y_val_age, prediction_val_age)
            
            
            result_XGB_splitset = {
                'n_estimator' : n_estimator,
                'learning_rate' : learning_rate,
                'max_depth' : max_depth,
                'mse_train' : mse_train,
                'mse_val' : mse_val
                
            }
            
            results_XGB_splitset.append(result_XGB_splitset)
results_df = pd.DataFrame(results_XGB_splitset)


# In[88]:


plt.title('Train x Validation metrics')
plt.xlabel('iteration')
plt.ylabel('MSE Age')
#plt.xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list) 
plt.plot(results_df['mse_train'])
plt.plot(results_df['mse_val'])
plt.legend(['Train','Validation'])


# In[89]:


best_params_train_XGB = results_df.loc[results_df['mse_train'].idxmin()]

print("Best parameters for minimum train MSE:")
print(best_params_train_XGB)

best_params_val_XGB = results_df.loc[results_df['mse_val'].idxmin()]

print("Best parameters for minimum validation MSE:")
print(best_params_val_XGB)


# In[90]:


model = XGBRegressor(n_estimators = 50, learning_rate = 1.5, max_depth = 10)
model.fit(X_train_age_full, y_train_age_full)

#model.predict(df[df["Age"].isna()][features_train])
predictions_train = model.predict(X_train_age_full)
mse_train = mean_squared_error(y_train_age_full, predictions_train)
print(f"The MSE of this model is {mse_train:.3f}")


# Assuming y_train and predictions_train are defined as your actual and predicted values
plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size
plt.scatter(y_train_age_full, predictions_train, alpha=0.5)  # Plots actual vs. predicted values
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Plot a diagonal line for reference
min_val = min(y_train_age_full.min(), predictions_train.min())
max_val = max(y_train_age_full.max(), predictions_train.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)  # 'k--' defines the style (black dashed line), lw is the line width

plt.show()


# In[93]:


best_model = XGBRegressor(n_estimators = 50, learning_rate = 1.5, max_depth = 10)
best_model.fit(X_train_age_full, y_train_age_full)


X_test_na = df[df["Age"].isna()][features_train]
X_test_na


# In[94]:


predicted_age = best_model.predict(X_test_na)


df.loc[df["Age"].isna(), "Age"] = predicted_age

df


# In[96]:


df["cabin_deck"] = df["Cabin"].str[0]
df[df['cabin_deck'].notna()]


# In[97]:


contingency_table = pd.crosstab(df['Pclass'], df['cabin_deck'])

# Stacked Bar Chart
contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Pclass Distribution across Cabin Decks')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Cabin Deck')
plt.show()


# In[98]:


df = pd.get_dummies(data = df, columns = ['cabin_deck'], dtype = int)
df


# In[99]:


features_to_train = ['SibSp','Parch',  'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Fare_norm']
features_target = ['cabin_deck_A', 'cabin_deck_B', 'cabin_deck_C', 'cabin_deck_D', 'cabin_deck_E', 'cabin_deck_F', 'cabin_deck_G']


# In[100]:


X_train_cabin_full = df[df["Cabin"].notna()][features_to_train]
y_train_cabin_full = df[df["Cabin"].notna()][features_target]


# In[101]:


model = XGBClassifier()
model.fit(X_train_cabin_full, y_train_cabin_full)
predictions_train  = model.predict(X_train_cabin_full)
accuracy_train = accuracy_score(predictions_train,y_train_cabin_full)


print(accuracy_train)


# In[102]:


X_train, X_val, y_train, y_val = train_test_split(X_train_cabin_full, y_train_cabin_full, train_size = 0.8)

n_estimators = [50,100,150,200,250,300,350,400,500]
learning_rates = [0.01, 0.03, 0.09, 0.27, 0.81]


# In[103]:


accuracy_list_train = []
accuracy_list_val = []

for n_estimator in n_estimators:
    for learning_rate in learning_rates:
        
        model =  XGBClassifier(n_estimators = n_estimator, learning_rate = learning_rate)
        model.fit(X_train, y_train)
        predictions_train = model.predict(X_train) 
        predictions_val = model.predict(X_val) 
        accuracy_train = accuracy_score(predictions_train,y_train)
        accuracy_val = accuracy_score(predictions_val,y_val)
        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])


# In[104]:


data = {
    'n_estimator': [n for n in n_estimators for lr in learning_rates],
    'learning_rate': learning_rates * len(n_estimators),
    'accuracy_train': accuracy_list_train,
    'accuracy_val': accuracy_list_val
}
results_df = pd.DataFrame(data)

# Plotting the results with learning rate values indicated on the x-axis
fig, axes = plt.subplots(len(n_estimators), 1, figsize=(10, 20), sharex=True)

for idx, n_estimator in enumerate(n_estimators):
    subset = results_df[results_df['n_estimator'] == n_estimator]
    axes[idx].plot(subset['learning_rate'], subset['accuracy_train'], marker='o', linestyle='-', label='Train')
    axes[idx].plot(subset['learning_rate'], subset['accuracy_val'], marker='x', linestyle='-', label='Validation')
    axes[idx].set_title(f'n_estimators: {n_estimator}')
    axes[idx].set_xlabel('Learning Rate')
    axes[idx].set_ylabel('Accuracy')
    axes[idx].legend()
    axes[idx].grid(True)
    # Setting the x-ticks to the learning rates
    axes[idx].set_xticks(learning_rates)
    axes[idx].set_xticklabels(learning_rates)

plt.tight_layout()
plt.show()


# In[105]:


best_model = XGBClassifier(n_estimators = 50, learning_rate = 0.81)

best_model.fit(X_train_cabin_full, y_train_cabin_full)
predictions_train  = best_model.predict(X_train_cabin_full)
accuracy_train = accuracy_score(predictions_train,y_train_cabin_full)


print(accuracy_train)


# In[106]:


predictions_test = best_model.predict(df[df['Cabin'].isna()][features_to_train])
predictions_test.shape


# In[107]:


indexer = df['Cabin'].isna()


df.loc[indexer, features_target] = predictions_test


# In[108]:


df.isna().sum()


# In[109]:


X_scaled_age = scaler.fit_transform(df[["Age"]].values)

df["Age_norm"] = X_scaled_age
df


# In[170]:


features_to_train = ['SibSp', 'Parch', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Fare_norm', 'cabin_deck_A', 'cabin_deck_B', 'cabin_deck_C', 'cabin_deck_D', 'cabin_deck_E', 'cabin_deck_F', 'cabin_deck_G', 'Age_norm']
target = ['Survived']

X_train_full = df[features_to_train]
y_train_full = df[target]

print(f"The shape of X train set is {X_train_full.shape}. The shape of y train set is {y_train_full.shape}")


# In[171]:


model_fullset = XGBClassifier()
model_fullset.fit(X_train_full, y_train_full)

prediction_train = model_fullset.predict(X_train_full)
accuracy_train = accuracy_score(prediction_train,y_train_full)

print(f"The accuracy of base model is {accuracy_train:.3f}")


# In[172]:


X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, train_size = 0.75)
print(f"The shape of X train and X val is {X_train.shape} and {X_val.shape}")
print(f"The shape of y train and y val is {y_train.shape} and {y_val.shape}")


# In[173]:


model_splitset = XGBClassifier()
model_splitset.fit(X_train, y_train)

prediction_train = model_splitset.predict(X_train)
accuracy_train = accuracy_score(prediction_train, y_train)

prediction_val = model_splitset.predict(X_val)
accuracy_val = accuracy_score(prediction_val, y_val)


print(f"Accuracy on training set is {accuracy_train:.2f}")
print(f"Accuracy on validation set is {accuracy_val:2f}")



# In[174]:


df_submission = pd.read_csv("C:/Users/kazja/OneDrive/Documents/Machine Learning Specalization/final_test_df.csv")
prediction_final = model_splitset.predict(df_submission)

print(f"The shape of final submission is {prediction_final.shape}")
prediction_final


                      
                        




