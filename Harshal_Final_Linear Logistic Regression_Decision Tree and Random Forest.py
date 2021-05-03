#!/usr/bin/env python
# coding: utf-8

# In[22]:


#%% Get the data
import pandas as pd

dm = pd.read_csv("C:/Users/Nehali/Downloads/creditcard.csv")


# In[23]:


#%% Discover and visualize the data to gain information
dm.head()


# In these five rows, based on the context of the problem, we can see that there are 28 main components (the V's), a time column (Time), a quantity attribute (Amount), which refers to the amount of the transaction and the column to predict (Class).
# 
# **Let's look at the information and description of this data set**

# In[24]:


dm.info()
cols = list(dm.columns)
dm.describe()


# The dataset does not contain empty values and, except for the Class column, which is of type integer, all other attributes are of type float. It is observed that the principal components have mean 0 and different standard deviations.
# 
# Let's now examine the Class variable:

# In[25]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor ='#E6E6E6', edgecolor = 'none', axisbelow = True, grid = True, prop_cycle = colors)
plt.rc('grid', color = 'w', linestyle = 'solid')
plt.rc('xtick', direction = 'out', color = 'black')
plt.rc('ytick', direction = 'out', color = 'black')
plt.rc('patch', edgecolor = '#E6E6E6')
plt.rc('lines', linewidth = 2)
plt.rcParams['figure.figsize'] = (11, 7)


# In[26]:


print(dm["Class"].value_counts())

plt.bar([0, 1], dm["Class"].value_counts())
plt.ylabel("Number of observations")
plt.title("Class")
plt.xticks([0, 1])
plt.legend()
plt.show()


# There is only 0.17% fraud. The imbalance is notoriously high!
# 
# Now let's take a look at the graphs of all the attributes in the dataset.

# In[27]:


plt.rcParams.update({'font.size': 18, 'figure.figsize': (44, 28)})
for l in range(len(dm.iloc[0, :])):
    plt.subplot(6, 6, l + 1)
    plt.hist(dm.iloc[:, l].values, bins = 50)
    plt.title(cols[l])
plt.show()


# In these graphs the values of the V's are centered on 0. On the other hand, some attributes follow a Gaussian distribution, if not all, since some attributes contain most of their values close to 0, so that their graph tends to be almost a line parallel to the y axis. Finally, the Amount graph, it seems that its values are centered at 0, but it is not. Rather, there are small quantities, but due to the scale, it is seen that they were 0, and there are quantities, counted with the fingers, exuberant. Remember that in this last attribute the minimum value is 88.34 and the maximum value of 25691.16, this last value is the consequence of the visualization.

# Out of curiosity, let's divide all the feature vectors corresponding to class 1 and calculate the modulus.

# In[28]:


# Separate class 1
plt.rcParams.update({'font.size': 12, 'figure.figsize': (11, 7)})

dm_C1 = dm.drop(dm[dm["Class"] == 0].index)

# Obtain module of each row (feature vector) for visualizate class
# dm_C1 behavior
mods = []
for i in range(len(dm_C1)):
    sum2 = 0
    for j in range(1, len(dm_C1.iloc[i, :]) - 2):
        sum2 = sum2 + dm_C1.iloc[i, j]**2
    mods.append(np.sqrt(sum2))
    
mods = pd.Series(mods)
mods.plot()
plt.title("Class 1")
plt.ylabel("Module of feature vector")
plt.xlabel("Number of feature vector")
plt.show();


# As only class 1 was separated, there are 492 feature vectors and, therefore, 492 modules. The modulus of a vector is the Euclidean norm.
# 
# **What can the vector modulus calculation work for?**
# 
# As we can see, it works to transform the exogenous variables (in this case only the V's attributes) into a series and, therefore, interesting patterns can be discovered. This also has its advantages because time analysis or neural network tools can then be applied to predict how many fraud there will be in the future! :). The downside at this point is that you have little data to apply a decent analysis. But the idea is what matters!
# 
# Let's now look at some correlations:

# In[29]:


# Looking for Correlations
import seaborn as sns
sns.set(font_scale = 1)

corr_matrix = dm.corr()
sns.heatmap(corr_matrix,
            xticklabels = corr_matrix.columns.values,
            yticklabels = corr_matrix.columns.values)
corr_matrix["Class"].sort_values(ascending = False)


# There are no strong linear correlations, it may be that there are other types of correlations. However, we will apply a classic data transformation: multiply some attributes and see if there is a stronger linear relationship.
# 
# The function from to under allows you to multiply the columns of a data frame.

# In[30]:


def attributeCombinationsCorr(df, colNameCorr, corrValLim):
    
    dataf = df.copy()
    cols = list(df.columns)
    types = [np.float64, np.int64]
    for j in range(len(df.iloc[0, :]) - 1):
        if type(df.iloc[0, j]) not in types:
            continue
        for jn in range(j + 1, len(df.iloc[0, :])):
            if type(df.iloc[0, jn]) not in types:
                continue
            dataf[cols[j] + cols[jn]] = df[cols[j]]*df[cols[jn]]
            if (len(df.iloc[0, :]) - j) == 2:
                break
    corr_matrix = dataf.corr()
    corrValues = corr_matrix[colNameCorr].sort_values(ascending = False)
    print(corrValues)
    
    corrIndexNames = list(corrValues.index.values)
    colsIndex = []
    for i in range(len(corrValues)):
        if corrValues[i] > corrValLim or corrValues[i] < -corrValLim:
            colsIndex.append(corrIndexNames[i])
    dfN = dataf[colsIndex]
    
    return dfN, corrValues


# In[31]:


# Using a function to combine attribute and increment its correlation

colNameCorr, corrValLim = "Class", 0.5
dfCorr, corrValues = attributeCombinationsCorr(dm, colNameCorr, corrValLim)

dfCorr = dfCorr.drop([name for name in list(dfCorr.columns)
                      if "Class" in name], axis = 1)
dfCorr = pd.concat([dfCorr, dm["Class"]], axis = 1)

corr_dfCorr = dfCorr.corr()
sns.heatmap(corr_dfCorr,
            xticklabels = corr_dfCorr.columns.values,
            yticklabels = corr_dfCorr.columns.values)
corr_dfCorr["Class"].sort_values(ascending = False)


# Linear correlations increased! It should be mentioned that multiplication with decimal numbers tends to decrease values. For this reason, the correlation with respect to the Class attribute increases. If we carry out multiplication with three attributes, the results will continue to decrease, increasing the correlation. That is why only binary multiplication was chosen.

# In[10]:


dfCorr.head()


# In[11]:


dfCorr.describe()


# **Let's move on to preparing the data for Machine Learning algorithms**
# 
# We have a new set of attributes with better linear correlation with respect to the Class attribute. In the next part a quantity of data is randomly extracted which is called "external_set". This data is never with the training set nor is it part of the test set. They are "new data" to verify the effectiveness of the algorithms. The training and test sets are obtained from the remaining set called "analysis_set".
# 
# Three classification algorithms are presented: Logistic Regression (LR), Decision Tree Classifier (DTC), and Random Forest Classifier (RFC). 

# In[12]:


#%% Prepare the data for Machine Learning algorithms.

np.random.seed(42)
def split_train_test(data, external_ratio):
    random_indices = np.random.permutation(len(data))
    external_set = int(len(data) * external_ratio)
    external_indices = random_indices[:external_set]
    analysis_set = random_indices[external_set:]
    return data.iloc[analysis_set], data.iloc[external_indices]

analysis_set, external_data = split_train_test(dfCorr, 0.1)

# Create a test and train sets of analysis_set
train, test = split_train_test(analysis_set, 0.2)

X_train, y_train = train.drop(["Class"], axis = 1), train["Class"]
X_test, y_test = test.drop(["Class"], axis = 1), test["Class"]

print("Train set \n", train["Class"].value_counts())
print("Test set \n", test["Class"].value_counts())
print("External set \n", external_data["Class"].value_counts())


# **Training a Linear Logistic Regression Model**

# In[13]:


#%% 5. Select a model and train it.

# Training a linear logistic regression model
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(class_weight = "balanced")
log_reg.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# To the evaluation of the Linear Logistic Regression model
fraud_pred = log_reg.predict(X_train)
print("Train set \n", confusion_matrix(y_train, fraud_pred))

from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1']
print(classification_report(y_train, fraud_pred, target_names = target_names, digits = 3))

class_names = [Class for Class in dm.Class.unique()]
plot_confusion_matrix(log_reg, X_train, y_train, display_labels = class_names, cmap = plt.cm.Reds)  
plt.show() 


# The trained logistic regression model presents a recall value slightly higher 0.90 for fraud detection.

# In[14]:


# Predictions test
y_pred = log_reg.predict(X_test)

print("Test set \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = target_names, digits = 3))

plot_confusion_matrix(log_reg, X_test, y_test, display_labels = class_names, cmap = plt.cm.Reds)
plt.show() 


# Focusing on fraud (most important for these problems) the present model gives a 0.88 recall with the test set.

# In[15]:


# Predictions external_data
outside_X = external_data.drop(["Class"], axis = 1)
outside_y = external_data["Class"]
y_pred_test = log_reg.predict(outside_X)
print("External set \n",confusion_matrix(outside_y, y_pred_test))
print(classification_report(outside_y, y_pred_test, target_names = target_names, digits = 3))

plot_confusion_matrix(log_reg, outside_X, outside_y, display_labels = class_names, cmap = plt.cm.Reds)
plt.show()


# Again, supposing that we take the logistic regression model as "good" and assign external data to it, it is observed that the recall presented for fraud detection is 0.89, which is very good. Only 5 errors had the model out of a total of 46 frauds.

# **Training a Decision Tree Classifier Model**

# In[16]:


# Train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

tree_Cla = DecisionTreeClassifier(class_weight = "balanced")
tree_Cla.fit(X_train, y_train)

fraud_pred = tree_Cla.predict(X_train)

print("Train set \n", confusion_matrix(y_train, fraud_pred))
print(classification_report(y_train, fraud_pred, target_names = target_names, digits = 3))

plot_confusion_matrix(tree_Cla, X_train, y_train, display_labels = class_names, cmap = plt.cm.Reds)  
plt.show()


# In this case, the data is overfitted in the model, that is, the machine learns in depth the behavior of the data, producing a perfect fit. However, since we have the test set and the outer set, it will be seen that the model does not perform good calculations.

# In[17]:


# Predictions test
y_pred = tree_Cla.predict(X_test)

print("Test set \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = target_names, digits = 3))

plot_confusion_matrix(tree_Cla, X_test, y_test, display_labels = class_names, cmap = plt.cm.Reds)  
plt.show()


# It is observed in the classification report that the recall has a value of 0.75 in the test set for fraud detection.

# In[18]:


# Predictions external_data
outside_X = external_data.drop(["Class"], axis = 1)
outside_y = external_data["Class"]
y_pred_test = tree_Cla.predict(outside_X)

print("External set \n", confusion_matrix(outside_y, y_pred_test))
print(classification_report(outside_y, y_pred_test, target_names = target_names, digits = 3))

plot_confusion_matrix(tree_Cla, outside_X, outside_y, display_labels = class_names, cmap = plt.cm.Reds)  
plt.show()


# In the latter, there was a larger decrease in the recall value compared to that obtained in the fraud detection test set.

# In[19]:


from sklearn.ensemble import RandomForestClassifier

forest_reg = RandomForestClassifier(class_weight = "balanced")
forest_reg.fit(X_train, y_train)

fraud_pred = forest_reg.predict(X_train)

print("Train set \n", confusion_matrix(y_train, fraud_pred))
print(classification_report(y_train, fraud_pred, target_names = target_names, digits = 3))

plot_confusion_matrix(forest_reg, X_train, y_train, display_labels = class_names, cmap = plt.cm.Reds)  
plt.show()


# The same thing happens here as in the case of the decision tree.

# In[20]:


# Predictions test
y_pred = forest_reg.predict(X_test)

print("Test set \n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = target_names, digits = 3))

plot_confusion_matrix(forest_reg, X_test, y_test, display_labels = class_names, cmap = plt.cm.Reds)  
plt.show()


# In this test set, the recall (for fraud) is greater than that obtained in the decision tree model but less than that obtained in the logistic regression.

# In[21]:


# Predictions external_data
outside_X = external_data.drop(["Class"], axis = 1)
outside_y = external_data["Class"]
y_pred_test = forest_reg.predict(outside_X)

print("External set \n", confusion_matrix(outside_y, y_pred_test))
print(classification_report(outside_y, y_pred_test, target_names = target_names, digits = 3))

plot_confusion_matrix(forest_reg, outside_X, outside_y, display_labels = class_names, cmap = plt.cm.Reds)  
plt.show()


# **Conclusion**
# 
# In this notebook a comparison of classification algorithms (Linear Logistic Regression, Decision Tree and Random Forest) was presented.
# * A function was created to carry out binary combinations (multiplication of attributes) by returning a dataframe with the combinations that meet the required value of correlation interest. In this case, those combinations that present a Pearson correlation greater than 0.5 were sealed.
# * From the principal components, binary combinations were carried out to improve the linear correlation with respect to the Class attribute.
# * Of the three classification models presented, the logistic regression model presents better results, secondly the random forest model and, lastly, the decision tree model.
# * In the logistic regression, a recall for fraud detection of 0.88 is obtained with the test set; a 0.89 recall with the outer set, and a 0.90 recall on training.
# 
# 
# I look forward to your feedback to keep improving in data science! :)
