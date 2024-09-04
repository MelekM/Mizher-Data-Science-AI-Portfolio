#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import library packages
import geopandas as gpd
import numpy as np
import statistics
import pandas as pd
import random

import calendar as cal
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS
from sklearn.base import clone

from pulp import LpVariable, LpProblem, LpMaximize, LpStatus, value, LpMinimize


# In[2]:


#Format function to show numbers more than 1000 in the ticks in K
def fmt_x(x,y):
    if x >= 1000000:        
        val = int(x) / 1000000
        return '{:.1f}M'.format(val)
    elif x >= 1000:        
        val = int(x) / 1000
        return '{val:d}K'.format(val=int(val))
    else:
        return int(x)
    


# ### Reading and understanding the construction data
# Essentially, the we want to derive the following based on our understanding from the 2020-2022 Austin construction dataset:
# * To identify the constraints affecting multifamily housing, e.g. number of units, impervious land, open space, number of rooms, land area, parking space, style, etc.
# * To create a linear model that quantitatively relates construction of multifamily housing with parameters mentioned above
# * To know the accuracy of the model, i.e. how well these variables can predict multifamily housing construction
# 

# In[3]:


# Construction data
constrct_data = pd.read_csv('/Users/rohantandon/Documents/NorthWestern University/MSDS 460 - Decision Analytics/Week 10/Final Project/Term-Project-Melek-Rohan-Spencer-Zain.csv')
constrct_data.head()


# In[4]:


# Shape
constrct_data.shape


# In[5]:


# Details on the attributes
constrct_data.info()


# In[6]:


# Keeping only constraints parameters
constrct_data_viz = constrct_data[['Number Of Units', 'Year Built', 'Height (Heighest BLD)',
                    'FAR', 'Impervious %', 'Building Coverage %', 'Open Space %', 
                    'Land Area (AC)', 'Parking Ratio']].copy()

# # Drop columns
# constrct_data_viz.drop(['FAR', 'Impervious %', 'Building Coverage %', 'Open Space %', 'Avg Unit SF', 
#         '% Studios', '% 1-Bed', '% 2-Bed', '% 3-Bed', '% 4-Bed', 'Closest Transit Stop Dist (mi)', 
#         'Land Area (AC)', 'Latitude', 'Longitude', 'Number Of 1 Bedrooms Units', 'Number Of 2 Bedrooms Units', 
#         'Number Of 3 Bedrooms Units', 'Number Of 4 Bedrooms Units', 'Number Of Studios Units', 'Parking Ratio'
#         ], axis=1, inplace=True)


# In[7]:


# Details of the updated attributes
constrct_data_viz.info()


# In[8]:


# Show modified data set
constrct_data_viz.head()


# In[9]:


# Describe dataset
constrct_data_viz.describe()


# ### Visualizing nummerical and categorical construction data
# Let us now visualize the construction data to understand the following:
# * If there is some obvious multicollinearity that exist between variables
# * To identify if some predictors directly have a strong association with the outcome variable
# 

# In[10]:


# Histogram of each parameter
constrct_data_viz.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()


# In[11]:


# Visualizing Numeric Variables in a form of pair plot
sns.pairplot(constrct_data_viz)
plt.figure(figsize=(30, 12))
plt.show()


# In[12]:


# Visualising Categorical Variables
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'Building Status', y = 'Number Of Units', data = constrct_data)
plt.subplot(2,3,2)
sns.boxplot(x = 'Style', y = 'Number Of Units', data = constrct_data)
plt.subplot(2,3,3)
sns.boxplot(x = 'County Name', y = 'Number Of Units', data = constrct_data)
plt.subplot(2,3,4)
sns.boxplot(x = 'Year Built', y = 'Number Of Units', data = constrct_data)
plt.subplot(2,3,5)
sns.boxplot(x = 'Number Of Stories', y = 'Number Of Units', data = constrct_data)
plt.subplot(2,3,6)
sns.boxplot(x = 'Submarket Cluster', y = 'Number Of Units', data = constrct_data)
plt.show()


# In[13]:


# Visualize categorical features
plt.figure(figsize = (14, 7))
plt.subplot(1,2,1)
sns.boxplot(x = 'Year Built', y = 'Number Of Units', hue = 'Building Status', data = constrct_data)
plt.subplot(1,2,2)
sns.boxplot(x = 'Year Built', y = 'Land Area (AC)', hue = 'Building Status', data = constrct_data)
plt.show()


# In[14]:


# Relationship between Number of Units built and Year Built
sns.jointplot(x = "Year Built", y = "Number Of Units",  
              data = constrct_data, kind='reg');


# In[15]:


# Getting median number of units by year
constrct_data_viz_splt1 = constrct_data_viz[['Year Built', 'Number Of Units', 'Land Area (AC)']].copy()
constrct_data_viz_splt1 = constrct_data_viz_splt1.groupby('Year Built', as_index=False).mean()
constrct_data_viz_splt1.head()


# In[16]:


# Plotting graph
ax=constrct_data_viz_splt1.plot(kind='line', x="Year Built", figsize=(15,8))
ax.yaxis.set_major_formatter(tick.FuncFormatter(fmt_x))
plt.xlabel("Year Built")
plt.ylabel("Median Count")
plt.title("Median volume of Units & Land Area approved over years (2020-2022) in Austin")
plt.show()


# In[17]:


# Getting median number of units by year
constrct_data_viz_splt2 = constrct_data_viz.copy()
constrct_data_viz_splt2 = constrct_data_viz_splt2.groupby('Year Built', as_index=False).mean()
constrct_data_viz_splt2.head()


# In[18]:


# Comparing all the constraints
sns.set_context("poster")
ax=constrct_data_viz_splt2.plot(kind='line', x='Year Built', figsize=(20,12), grid=True)
ax.yaxis.set_major_formatter(tick.FuncFormatter(fmt_x))
plt.xlabel("Year Built")
plt.ylabel("Metrics Volume")
plt.title("2020-2022 Exisiting and New Construction near Subject Property (5-mile radius)")
plt.show()


# ### Correlation Matrix to show correlation coefficients of variables that are highly correlated

# In[19]:


# Correlation Matrix
plt.figure(figsize = (16, 10))
sns.heatmap(constrct_data_viz.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[20]:


# Mapping Longitude and Latitude
constrct_data.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.5, figsize=(10,7),
    c="Number Of Units", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)


# In[21]:


# Separating out the features based on correlation matrix
attributes = ['Number Of Units', 'Height (Heighest BLD)', 'FAR', 'Impervious %', 'Building Coverage %', 
            'Open Space %', 'Parking Ratio']

# creating a copy
constrct_data_lp = constrct_data[attributes].copy()
constrct_data_lp = constrct_data_lp.dropna(axis=1, how='all')
constrct_data_lp = constrct_data_lp.dropna(axis=0, how='all')
constrct_data_lp


# In[22]:


# Separate Target from other attributes
y = constrct_data_lp['Number Of Units'].to_numpy(copy=True)
constrct_datamod = constrct_data_lp.drop(['Number Of Units'], axis=1, inplace=False)
features = constrct_datamod.columns.to_list()
x = constrct_datamod.loc[:,features].to_numpy(copy=True)


# In[23]:


trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.15, random_state=77) 


# In[24]:


# Randon Forest Regression
RFregr=RandomForestRegressor(n_estimators=200, max_features='log2',oob_score=True,n_jobs=-1, random_state=11)
RFregr.fit(trainX,trainy)


# In[25]:


# Printing RF R Sq
print(f'RF R Squared, Training: {RFregr.score(trainX,trainy):5.3f}')


# In[26]:


# Below provides a % breakdown on feature importance to predicting the # of multifamily units. 
# The importance percentages will be used as the basis for the weights for each Lpvariable proxy 
RFFeatImpDF=pd.DataFrame({'feature':features,'importance':RFregr.feature_importances_})
print('Feature importances')
RFFeatImpDF.sort_values('importance',ascending=False)


# In[27]:


# define variables
HP = LpVariable("Height Proxy", 0, None) # Height proxy is the transformation variabe used to convert height and land acre area to # of units
FP = LpVariable("FAR Proxy", 0, None) # FAR proxy is the transformation variabe used to convert height and land acre area to # of units
IP = LpVariable("Impervious Coverage Proxy", 0, None) # Impervious Coverage proxy is the transformation variabe used to convert impervious coverage and land acre area to # of units
BP = LpVariable("Building Coverage Proxy", 0, None) # Building Coverage proxy is the transformation variabe used to convert Building Coverage and land acre area to # of units
OP = LpVariable("Open Space Proxy", 0, None) # Open Space proxy is the transformation variabe used to convert Open Space and land acre area to # of units
PP = LpVariable("Parking Ratio Proxy", 0, None) # Parking proxy is the transformation variabe used to convert Parking Ratio and land acre area to # of units

# defines the problem
prob = LpProblem("problem", LpMaximize)

# Define Height constraints
prob += HP >= 0.4 # Lowest height proxy (units/height/land acreage) of sample multifamily properties
prob += HP <= 1.20 # Highest height proxy (units/height/land acreage) of sample multifamily properties
prob += 50.18*HP*10.40 <= 263.85 # Height proxy multiplied by average sample height and average land acre must be at or below the average number of units
# Height proxy constraint indicating the height proxy level must be limited at or below the total units size of the sample set, ceofficents are height * land acreage 
prob += 864.03*HP + 428.80*HP + 724.51*HP + 490.72*HP + 582.08*HP + 209.48*HP + 677.71*HP + 252.44*HP + 566.96*HP + 485.10*HP + 179.52*HP + 134.64*HP <= 2880

# Define FAR (Floor to Area Ratio) constraints
prob += FP >= 13.2 # Lowest FAR proxy (units/FAR/land acreage) of sample multifamily properties
prob += FP <= 95.7 # Highest FAR proxy (units/FAR/land acreage) of sample multifamily properties
prob += 0.74*FP*10.40 <= 263.85 # FAR proxy multiplied by average sample FAR and average land acre must be at or below the average number of units
# FAR proxy constraint indicating the FAR proxy level must be limited at or below the total units size of the sample set, ceofficents are FAR * land acreage 
prob += 9.66*FP + 5.15*FP + 8.07*FP + 7.75*FP + 3.94*FP + 3.25*FP + 9.02*FP + 4.23*FP + 7.48*FP + 14.96*FP + 5.39*FP + 4.01*FP <= 2880

# Define Impervious Coverage % constraints
prob += IP >= 29.3 # Lowest Impervious Coverage % proxy (units/impervious cover %/land acreage) of sample multifamily properties
prob += IP <= 73.1 # Highest Impervious Coverage % proxy (units/impervious cover %/land acreage) of sample multifamily properties
prob += 0.61*IP*10.40 <= 263.85 # Impervious coverage % proxy multiplied by average sample impervious coverage % and average land acre must be at or below the average number of units
# Impervious Coverage % proxy constraint indicating the Impervious Coverage % proxy level must be limited at or below the total units size of the sample set, ceofficents are FAR * land acreage 
prob += 8.65*IP + 5.93*IP + 6.67*IP + 7.75*IP + 7.11*IP + 2.57*IP + 7.13*IP + 3.47*IP + 6.60*IP + 4.12*IP + 2.91*IP + 2.28*IP <= 2880

# Define Building Coverage % constraints
prob += BP >= 78.0 # Lowest Building Coverage % proxy (units/Building Coverage %/land acreage) of sample multifamily properties
prob += BP <= 158.2 # Highest Building Coverage % proxy (units/Building Coverage %/land acreage) of sample multifamily properties
prob += 0.27*BP*10.40 <= 263.85 # Building Coverage % proxy multiplied by average sample Building Coverage % and average land acre must be at or below the average number of units
# Building Coverage % proxy constraint indicating the Building Coverage % proxy level must be limited at or below the total units size of the sample set, ceofficents are FAR * land acreage 
prob += 2.87*BP + 1.24*BP + 2.23*BP + 2.98*BP + 3.59*BP + 0.93*BP + 3.46*BP + 1.07*BP + 2.18*BP + 1.72*BP + 2.03*BP + 1.61*BP <= 2880

# Define Open Space % constraints
prob += OP >= 0.0 # Lowest Open Space % proxy (units/Open Space %/land acreage) of sample multifamily properties
prob += OP <= 497.2 # Highest Open Space % proxy (units/Open Space %/land acreage) of sample multifamily properties
prob += 0.04*OP*10.40 <= 263.85 # Open Space % proxy multiplied by average sample Open Space % and average land acre must be at or below the average number of units
# Open Space % proxy constraint indicating the Open Space % proxy level must be limited at or below the total units size of the sample set, ceofficents are FAR * land acreage 
prob += 2.65*OP + 0.56*OP + 0.61*OP + 0.82*OP + 0.57*OP + 0.70*OP <= 2880

# Define Parking Ratio constraints
prob += PP >= 10.1 # Lowest Parking Ratio proxy (units/Parking Ratio/land acreage) of sample multifamily properties
prob += PP <= 33.0 # Highest Parking Ratio proxy (units/Parking Ratio/land acreage) of sample multifamily properties
prob += 1.57*OP*10.40 <= 263.85 # Parking Ratio proxy multiplied by average sample Parking Ratio and average land acre must be at or below the average number of units
# Parking Ratio proxy constraint indicating the Parking Ratio proxy level must be limited at or below the total units size of the sample set, ceofficents are FAR * land acreage 
prob += 26.88*PP + 11.03*PP + 20.15*PP + 22.53*PP + 13.31*PP + 4.46*PP + 21.44*PP + 10.41*PP + 18.31*PP + 19.53*PP + 6.54*PP + 5.01*PP <= 2880

# define objective function

LA = 9.654 # Subject land acreage

# Weighting is based on feature importance to # of units per the Random Forest Regressor
prob += 0.243721*(HP*LA) + 0.188389*(FP*LA) + 0.135877*(IP*LA) + 0.135074*(BP*LA) + 0.059508*(OP*LA) + 0.059508*(OP*LA) + 0.237432*(PP*LA)      

# solve the problem
status = prob.solve()
print(f"First LP")
print(f"status={LpStatus[status]}")

# print the results
for variable in prob.variables():
    print(f"{variable.name} = {variable.varValue}")
    
print(f"Optimal Number of Units = {value(prob.objective)}")
print(f"")

