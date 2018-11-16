
# coding: utf-8

# # Dimensionality Reduction and Clustering of UK Biobank Asthma Patients
# 
# This notebook does the following:
# 
# * Reads in a sample of UKBB Asthma patients
# * Performs dimensionality reduction on UKBB Asthma Patients with PCA and finds the optimal number of principal components
# * Performs dimensionality reduction on UKBB Asthma patients wit tSNE, using the otpimal number of principal components
# * Performs kMeans clustering to find any clusters of UKBB asthma patients
# 
# 

# In[7]:

import pandas as pd 
import numpy as np
get_ipython().magic(u'matplotlib inline')
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(12,6)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn import decomposition
from sklearn import manifold
from sklearn.cluster import KMeans 


# # Step 1: Read in UKBB asthma patients and subselect data
# 
# Fields in UKBB phenotype data come in three types:
# 
# 1) 'QUANTITY' fields - these fields are continuous variables. Integers and Decimal numbers
# 2) 'CATEGORY' fields - these fields are not continuous. There are a limited number of integers which label specfic category values
# 3) 'BIN' fields - these are binary fields. They only have two possible values. They only label a field as being 'TRUE' or 'FALSE' 
# 
# For summary statistics and finding the most correlated variables, we only selected 'QUANTITY' fields.  For dimensionality reduction for further clustering, it's not so clear whether we have to select only 'QUANTITY' fields.

# In[8]:

#Read in 5K sample
#asthma_df=pd.read_csv("ukbb_asthma_sample_5k.csv",sep='\t')
#asthma_df=pd.read_csv("ukbb_asthma_sample_5k_new.csv",sep='\t')


# In[9]:

asthma_df=pd.read_csv("ukbb_asthma_full_filtered.csv",sep='\t')


# In[10]:

asthma_df.columns


# In[11]:

# Select columns to filter on
cols = [col for col in asthma_df.columns if 'HES_p' not in col and 'PC' not in col]


# In[12]:

#asthma_df=asthma_df[cols].drop('Unnamed: 0',axis=1)


# In[13]:

#select only 'QUANTITY' fields
asthma_df_quant=asthma_df.loc[:, asthma_df.columns.str.contains('QUANT|age|BMI')].fillna(0.0)
#select only fields with 10 or more unique values
for col in asthma_df_quant.columns:
    if len(asthma_df_quant[col].unique()) < 20 :
        asthma_df_quant.drop(col,inplace=True,axis=1)

        
#select only 'QUANTITY' and "CATEGORY" fields
#asthma_df_quant_cat=asthma_df.loc[:, asthma_df.columns.str.contains('QUANT|age|BMI|CAT|sex')].fillna(0.0)
#asthma_df_quant_cat=asthma_df_quant_cat.drop(columns=['f_22182_0_0_f_CAT_HLA_imputation_values_and_quality'])
#selects all features. 

#asthma_df_all=asthma_df.loc[:,~asthma_df.columns.str.contains('ID|random|chip|asthma')].fillna(0.0)
#asthma_df_all=asthma_df_all.drop(columns=['f_22182_0_0_f_CAT_HLA_imputation_values_and_quality'])


# In[14]:

asthma_df_quant.shape


# # Step 2: Reduce dimensions with PCA
# 
# The next step is to perform PCA to reduce the dimensions in the data set. PCA is straightforward with Python. It's not straightforward to identify the ideal number of Principal Components to use and which data to include. We have three data frames in this notebook:
# 
# * asthma_df_quant - this data frame only has 'QUANTITY' fields. This is exaclty the same as the exploratory analysis phase finding summary statistics and correlated/uncorrelated variables.
# * asthma_df_quant_cat - this data frame has both 'QUANTITY' fields and 'CATEGORY' fields. This totals about 1500 features
# * asthma_df_all - this data frame has everything - 'QUANTITY', 'CATEGORY', and 'BINARY' fields.
# 
# 
# In this step we have to:
# 
# * Perform PCA in two configurations: Two principal components and the optimal number of principal components
# * We need to perform this PCA on all three data frames listed above. 
# 
# 

# ## Data Scaling
# Before dimensionality reduction and further clustering with kMeans, data has to be scaled. The standard scaler scales the data to an average value of 0 and a unit variance of 1. This is appropriate for the 'QUANTITY' features. We can't use the standard scaler for the other features, so we do 'min/max' scaling, which scales the data in the range from 0 to 1.
# 

# In[15]:

#Scale the quantity features only
data_quant=StandardScaler().fit_transform(asthma_df_quant.values)
#Scale the quantity AND CATEGORY features
#data_quant_cat=StandardScaler().fit_transform(asthma_df_quant_cat.applymap(lambda x: float(x)).values)
#Scale ALL THE features
#data_all=StandardScaler().fit_transform(asthma_df_all.applymap(lambda x:float(x)).values)


# ## PCA 

# In[16]:

pca_2d = decomposition.PCA(n_components=2)
pca_quant_2d=pca_2d.fit_transform(data_quant)

plt.scatter(pca_quant_2d[:,0],pca_quant_2d[:,1])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('QUANTITY FEATURES ONLY two principal components')
plt.show()


# ## Finding the optimal number of principal components for each three data frames
# 
# It's easier to talk about in person, but the best way to determine the optimal number of principal components is to plot the 'cummulative eigenvalue fraction' vs the number of eigenvalues. The code to do this is below. The example code is for 500 principal components - this is way too many. I'll explain why below:
# 

# In[17]:

# PCA for 500 components
pca_500d = decomposition.PCA(n_components=500)
pca_quant_500d=pca_500d.fit_transform(data_quant)

# Calculate the cummalative eigenvalue fraction
eigen_fraction_500d=np.cumsum(pca_500d.explained_variance_)/(np.sum(pca_500d.explained_variance_))
#Plot the Results
plt.scatter(range(1,len(pca_500d.explained_variance_)+1),eigen_fraction_500d)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cummatlive Eigenvalue Fraction')
plt.title('QUANTITY FEATURES ONLY 500 principal components')
plt.show()


# ## Do you see how the plot is flat for most of it? It means 500 is way too many principal components. The optimal number of Principal components is when the above plot only starts to get flat. Copy and paste the code below to find the optimal number of Principal Components for all three data frames:
# 
# * Quantitative Features Only
# * Quantitative + Categorical Features
# * All Features
# 
# #### Hint
# 
# This is not as hard as it looks. You can reduce the number of principal components in the above plot (to say 150 or so) and pick out the optimal number of components by eye. 

# In[18]:

# PCA for 300 components

pca_300d = decomposition.PCA(n_components=300)
pca_quant_300d=pca_300d.fit_transform(data_quant)

# Calculate the cummalative eigenvalue fraction
eigen_fraction_300d=np.cumsum(pca_300d.explained_variance_)/(np.sum(pca_300d.explained_variance_))
#Plot the Results
plt.scatter(range(1,len(pca_300d.explained_variance_)+1),eigen_fraction_300d)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cummatlive Eigenvalue Fraction')
plt.xlim(5,60)
plt.ylim(0.1,0.8)
plt.title('QUANTITY FEATURES ONLY 300 principal components')
plt.show()


# In[19]:

pca_55d = decomposition.PCA(n_components=55)
pca_quant_55d=pca_55d.fit_transform(data_quant)

plt.scatter(pca_quant_55d[:,0],pca_quant_55d[:,1])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('QUANTITY FEATURES ONLY 55 principal components')
plt.show()


# # Step 3: Dimensionality Reduction with tSNE
# 
# This part is straightforward. You will have to run the same code below and replace the number of dimensions with the optimal number of principal components you found above. You will have to do so again for all three data frames

# In[20]:

'''tsne_2d = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,
                     learning_rate=300, n_iter=400)
    #tsne = manifold.TSNE(n_components=2, init='pca')
tsne_quant_2d=tsne_2d.fit_transform(data_quant)

plt.scatter(tsne_quant_2d[:,0],tsne_quant_2d[:,1])

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('QUANTITY FEATURES ONLY tSNE with two components')
plt.show()
'''


# In[21]:

"""tsne_2d = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,
                     learning_rate=300, n_iter=400)
    #tsne = manifold.TSNE(n_components=2, init='pca')
tsne_quant_2d=tsne_2d.fit_transform(pca_quant_55d)

plt.scatter(tsne_quant_2d[:,0],tsne_quant_2d[:,1])

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('QUANTITY FEATURES ONLY tSNE with two components')
plt.show()
"""


# # Step 4 - kMeans Clustering
# 
# We will have to do kMeans clustering on the data frame with quantitative features only, on the PCA transformed data. We will do kMeans on two different inputs:
# 
# * Two principal components 
# * The optimal number of principal components (55)
# 
# 
# ## 2 principal component kMeans
# 
# From the PCA plots above, it looks like there are three clusters, possibly 4. A tSNE visualization on the PCA output would give us a better idea how many clusters there are likely to be. To perform kMeans clustering, you must perform the following steps:
# 
# * Run kMeans with the parameters n_init=15 and max_iter=400 on the 2-dimensional PCA output to find 3 clusters. Look at the documentation in http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html and https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
# * Visualize the resulsting clusters on the 2-dimensional PCA scatter plot above. We want to just reproduce that plot and color the points with which cluster they belong to. Cell #4 in https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html demonstrates how to do this
# * We next have to optimize the number of clusters. There are many ways to do this, but using the Silhouette Score to select the best number of clusters is probably the most direct and the fastest: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html and http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score Basically, we want the number of clusters such that the silhouette score (a measure of cluster separation) is as close to 1.0 as possible. Start with 2 clusters, Don't go over 8 clusters. 
# 
# ## Optimal number (55) principal components
# 
# Repeat the above steps, but with 55 principal components. You can't visualize 55 dimensions of course, so just plot the first two principal components of the 55. 

# In[22]:

X = pca_quant_55d
kmeans = KMeans(n_clusters=2, n_init=15, max_iter=400).fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c=y_pred)


# In[23]:

len(y_pred)


# In[24]:

X = pca_quant_55d
kmeans = KMeans(n_clusters=2, n_init=15, max_iter=400)
from sklearn.metrics import silhouette_samples, silhouette_score
cluster_labels = kmeans.fit_predict(X)

silhouette_avg = silhouette_score(X, cluster_labels)
print(silhouette_avg)
plt.scatter(X[:,0], X[:,1], c=cluster_labels, cmap="coolwarm")
plt.title("Kmeans Clustering for 55 Principal Components")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
#plt.savefig('kmeans_55d_scatterplot.png')


# from sklearn.cluster import AgglomerativeClustering
# from sklearn.cluster import MeanShift, estimate_bandwidth
# 
# X = pca_quant_55d
# n=2
# #clusterer = KMeans(n_clusters=n, random_state=1
# clusterer = AgglomerativeClustering(n_clusters=n,linkage='average',affinity='l1')
# 
# cluster_labels = clusterer.fit_predict(X)
# 
# silhouette_avg = silhouette_score(X, cluster_labels)
# print(silhouette_avg)
# plt.scatter(X[:,0], X[:,1], c=cluster_labels)
# plt.show()

# # Step 5 - find which features contribute the most to PCA. 

# In[25]:

## which variables change the data the most/ have the highest variance 


# In[26]:

import seaborn as sns
plt.rcParams['figure.figsize']=(12,6)

sns.heatmap(np.log(pca_55d.inverse_transform(np.eye(pca_quant_55d.shape[1])))[:,0:100], cmap="hot", cbar=True)
plt.ylabel('Principal Component')
plt.xlabel('Quantitative Feature index')
plt.show()


# In[27]:

## pale colors are most important 


# ### The above plot shows a heat map of feature importances. Because we have over 600 quantitative features, I only show 100. There are 55 principal components. The more yellow the heatmap is, the more important the feature is. Let's average over the principal components to get the most important features

# In[28]:

log_importances=np.log(pca_55d.inverse_transform(np.eye(pca_quant_55d.shape[1])))


# In[29]:

mean_log_importances=np.apply_along_axis(np.mean,0,np.nan_to_num(log_importances))
print(len(mean_log_importances))


# In[30]:

mean_log_importances_df=pd.DataFrame(data=mean_log_importances,columns=['importance'])


# In[31]:

mean_log_importances_df=mean_log_importances_df.reset_index()
print(mean_log_importances_df.head(15))


# ### The numpy array above is an array of importances, of length 638. They map back to the original data frame:

# In[32]:

print(asthma_df_quant.columns[0],asthma_df_quant.columns[637])


# In[33]:

## every entry in array, can get variabe name 


# # Find the ten most important features to start, including their names, to focus on. 

# In[34]:

## labels 0-638: have value associated. 
## output array as csv file -- find 10 highest values in the array and associated array index number 

np.savetxt("Asthma Important Features.csv", mean_log_importances_df, delimiter=",")


# In[35]:

asthma_df_important_features = mean_log_importances_df.sort_values(by="importance", ascending=False)
print(asthma_df_important_features.head(15))


# # Step 6: ANOVA of the ten most important features between the two clusters

# In[36]:

# Add the group label to the data frame:
asthma_df_quant['Cluster Label']=y_pred
asthma_df_quant['Cluster Label']=asthma_df_quant['Cluster Label'].apply(str).apply(lambda x: x.replace('0','A')).apply(lambda x: x.replace('1','B'))


# In[37]:

asthma_df_quant.shape


# ### adding the cluster labels allows you to visualize box plots broken down by cluster easily. For example, for age:

# In[38]:

## include box plots for top 10 variables 


# In[39]:

asthma_df_quant.boxplot(asthma_df_quant.columns[0],by='Cluster Label')


# In[40]:

v = asthma_df_important_features['index'].values[0]


# In[41]:

asthma_df_quant.boxplot(asthma_df_quant.columns[449],by='Cluster Label')


# In[42]:

asthma_df_quant.boxplot(asthma_df_quant.columns[584],by='Cluster Label')


# In[43]:

asthma_df_quant.boxplot(asthma_df_quant.columns[659],by='Cluster Label')


# In[44]:

asthma_df_quant.boxplot(asthma_df_quant.columns[617],by='Cluster Label')


# In[45]:

asthma_df_quant.boxplot(asthma_df_quant.columns[453],by='Cluster Label')


# In[46]:

asthma_df_quant.boxplot(asthma_df_quant.columns[357],by='Cluster Label')


# In[47]:

asthma_df_quant.boxplot(asthma_df_quant.columns[10],by='Cluster Label')


# In[48]:

asthma_df_quant.boxplot(asthma_df_quant.columns[597],by='Cluster Label')


# In[49]:

asthma_df_quant.boxplot(asthma_df_quant.columns[411],by='Cluster Label')


# In[50]:

asthma_df_quant.boxplot(asthma_df_quant.columns[612],by='Cluster Label')


# In[51]:

asthma_df_quant.boxplot(asthma_df_quant.columns[109],by='Cluster Label')


# In[52]:

asthma_df_quant.boxplot(asthma_df_quant.columns[605],by='Cluster Label')


# In[53]:

asthma_df_quant.boxplot(asthma_df_quant.columns[56],by='Cluster Label')


# In[54]:

asthma_df_quant.boxplot(asthma_df_quant.columns[603],by='Cluster Label')


# In[55]:

asthma_df_quant.boxplot(asthma_df_quant.columns[60],by='Cluster Label')


# ### which means the Data is not clustering by age. 
# 
# 1) Visualize the data in box plots for all ten most important features
# 2) Run ANOVA on the ten most important features above to derive p-values. Good resources for ANOVA in Python include: https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/ (first section with SciPy only) 
# 
# and
# 
# https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.f_oneway.html
# 
# 

# # ANOVA - analyzes groups of data (group A vs B) to quantify how different they are 
# ## do each variable separately - 15 different ANOVA's
# ## lower the better p-value. we define the cutoff ( < 0.05)

# In[59]:

asthma_df_quant_group1 = asthma_df_quant[asthma_df_quant['Cluster Label'].str.contains('A')]

asthma_df_quant_group2 = asthma_df_quant[asthma_df_quant['Cluster Label'].str.contains('B')]


# In[60]:

[asthma_df_quant.columns[381]]


# In[61]:

import scipy.stats as stats

feature = asthma_df_quant.columns[449]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))
## if pvalue < 0.05:

   


# In[62]:

feature = asthma_df_quant.columns[584]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[63]:

feature = asthma_df_quant.columns[659]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[64]:

feature = asthma_df_quant.columns[617]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[65]:

[asthma_df_quant.columns[453]]


# In[66]:

feature = asthma_df_quant.columns[453]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[67]:

feature = asthma_df_quant.columns[357]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[68]:

feature = asthma_df_quant.columns[10]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[69]:

feature = asthma_df_quant.columns[597]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[70]:

[asthma_df_quant.columns[411]]


# In[71]:

feature = asthma_df_quant.columns[411]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[72]:

[asthma_df_quant.columns[612]]


# In[73]:

feature = asthma_df_quant.columns[612]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[74]:

feature = asthma_df_quant.columns[109]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[75]:

feature = asthma_df_quant.columns[605]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[76]:

feature = asthma_df_quant.columns[56]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[77]:

feature = asthma_df_quant.columns[603]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[78]:

feature = asthma_df_quant.columns[60]
(statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    
print("ANOVA Statistic " + str(statistic) + " and p-value " + str(pvalue))


# In[79]:

asthma_df.columns


# In[80]:

p_values=[]
features=list(asthma_df_quant_group1.columns.values[:-1])
anova_dict=dict()

for feature in asthma_df_quant_group1.columns.values[:-1]:
    (statistic, pvalue) = stats.f_oneway(asthma_df_quant_group1[feature].values, asthma_df_quant_group2[feature].values)
    p_values.append(pvalue)
anova_dict['feature']=features
anova_dict['pvalue']=p_values
anova_df=pd.DataFrame.from_dict(anova_dict)


# In[81]:

anova_df=anova_df.sort_values(by='pvalue')
threshold=0.01
anova_df_significant=anova_df.loc[(anova_df.pvalue < threshold) & (anova_df.pvalue > 0)].sort_values(by='pvalue')


# In[82]:

anova_df_significant.tail(50)


# In[83]:

anova_df_significant[anova_df_significant['feature'].str.contains('FEV|FVC')].shape


# In[84]:

anova_df_significant.shape


# In[85]:

asthma_df_quant.boxplot(asthma_df_quant.columns[601],by='Cluster Label')


# In[86]:

asthma_df_quant.boxplot(asthma_df_quant.columns[270],by='Cluster Label')


# In[87]:

print('hello')


# ## Features that are lung function-specific

# In[88]:

anova_df_lung=anova_df_significant[anova_df_significant['feature'].str.contains('FEV|FVC|cough|smok|asthma|tobacco|bronc|Lung|lung|PEF')]


# In[89]:

anova_df_lung.shape


# In[ ]:




# In[90]:

anova_df_lung.head(20)


# In[91]:

asthma_df_important_features_lung=asthma_df_important_features[asthma_df_important_features['index'].isin(anova_df_lung.index)]


# In[92]:

asthma_df_important_features_lung.head(10)


# In[93]:

asthma_df_important_features_lung.shape


# In[94]:

asthma_df_quant.boxplot(asthma_df_quant.columns[358],by='Cluster Label')


# In[95]:

asthma_df_quant.boxplot(asthma_df_quant.columns[652],by='Cluster Label')


# In[96]:

asthma_df_quant.boxplot(asthma_df_quant.columns[470],by='Cluster Label')


# In[97]:

asthma_df_quant.boxplot(asthma_df_quant.columns[92],by='Cluster Label')


# In[98]:

asthma_df_quant.boxplot(asthma_df_quant.columns[654],by='Cluster Label')


# ## Features that are blood specific

# In[99]:

anova_df_blood=anova_df_significant[anova_df_significant['feature'].str.contains('LV|cardiac|Neutrophil|blood|Basophil|Platelet|Pulse|Heart|Cardiac|monocyte|heart|Reticulocyte')]


# In[100]:

anova_df_blood.shape


# In[101]:

anova_df_blood


# In[102]:

asthma_df_important_features_blood=asthma_df_important_features[asthma_df_important_features['index'].isin(anova_df_blood.index)]


# In[103]:

asthma_df_important_features_blood.head(10)


# In[104]:

asthma_df_quant.boxplot(asthma_df_quant.columns[604],by='Cluster Label')


# In[105]:

asthma_df_quant.boxplot(asthma_df_quant.columns[605],by='Cluster Label')


# In[106]:

asthma_df_quant.boxplot(asthma_df_quant.columns[621],by='Cluster Label')


# In[107]:

asthma_df_quant.boxplot(asthma_df_quant.columns[612],by='Cluster Label')


# In[108]:

asthma_df_quant.boxplot(asthma_df_quant.columns[483],by='Cluster Label')


# ## Other features that are present in healthy patients

# In[109]:

anova_df_healthy=anova_df_significant[anova_df_significant['feature'].str.contains('fat|bone|mass|grip|adipose|thigh|Waist|BMI|Hip|Ankle|Corneal|ocular')]


# In[110]:

anova_df_healthy.shape


# In[111]:

anova_df_healthy.head(20)


# In[112]:

asthma_df_quant.boxplot(asthma_df_quant.columns[538],by='Cluster Label')


# In[113]:

asthma_df_quant.boxplot(asthma_df_quant.columns[559],by='Cluster Label')


# In[114]:

asthma_df_quant.boxplot(asthma_df_quant.columns[546],by='Cluster Label')


# In[115]:

asthma_df_quant.boxplot(asthma_df_quant.columns[555],by='Cluster Label')


# In[116]:

asthma_df_quant.boxplot(asthma_df_quant.columns[585],by='Cluster Label')


# ## Other features that may be specific to asthma

# In[117]:

anova_df_other=anova_df_significant[~anova_df_significant['feature'].str.contains('fat|bone|mass|grip|adipose|thigh|Waist|BMI|Hip|Ankle|Corneal|ocular|LV|cardiac|Neutrophil|blood|Basophil|Platelet|Pulse|Heart|Cardiac|monocyte|heart|Reticulocyte|FEV|FVC|cough|smok|asthma|tobacco|bronc|Lung|lung|PEF')]


# In[118]:

anova_df_other.shape


# In[119]:

anova_df_other.head(10)


# In[120]:

asthma_df_important_features_other=asthma_df_important_features[asthma_df_important_features['index'].isin(anova_df_other.index)]


# In[121]:

asthma_df_important_features_other.head(10)


# ## Step 7: Write out Asthma Group Labels for different cohorts 

# In[138]:

cohort_labels_df=pd.get_dummies(asthma_df_quant['Cluster Label']).rename(columns={'A':'Asthma_CohortA','B':'Asthma_CohortB'})
cohort_labels_df['Asthma_CohortAnotB']=cohort_labels_df['Asthma_CohortA'] 


# In[139]:

cohort_labels_df.head()


# In[152]:

asthma_df_full_gwas=pd.read_csv('ukbb_asthma_full.csv',sep='\t',usecols=['IID','PC1','PC2','PC3','PC4','PC5','PC6','PC7',
                                                           'PC8','PC9','PC10','sex','age'])


# In[153]:

asthma_df_full_gwas.head()


# In[154]:

cohort_labels_df.head()


# In[172]:

asthma_df_full_gwas=asthma_df_full_gwas.join(cohort_labels_df)
asthma_df_full_gwas.to_csv('ukbb_asthma_cohort_annotation.csv')


# In[165]:

asthma_df_full_not_asthma=pd.read_csv('ukbb_not_asthma_full_gwas.csv',sep='\t').drop(columns=['Unnamed: 0'])


# In[168]:

asthma_df_full_not_asthma['Asthma_CohortA']=0
asthma_df_full_not_asthma['Asthma_CohortB']=0
asthma_df_full_not_asthma['Asthma_CohortAnotB']=np.NaN


# In[193]:

ukbb_df_full_gwas=pd.concat([asthma_df_full_gwas,asthma_df_full_not_asthma]).reset_index().drop(columns='index')


# In[194]:

ukbb_df_full_gwas=ukbb_df_full_gwas[['IID','PC1','PC2','PC3','PC4','PC5','PC6','PC7',
'PC8','PC9','PC10','sex','age','Asthma_CohortA','Asthma_CohortB','Asthma_CohortAnotB']]


# In[196]:

ukbb_df_full_gwas.to_csv('ukbb_asthma_cohorts_gwas_annotation.csv',index=False)


# In[ ]:



