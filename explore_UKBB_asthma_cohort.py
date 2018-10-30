
# coding: utf-8

# In[79]:

import pandas as pd 
import numpy as np
get_ipython().magic(u'matplotlib inline')
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(12,6)


# In[18]:

#Read in 5K sample
asthma_df=pd.read_csv("ukbb_asthma_sample.csv",sep='\t')


# In[19]:

asthma_df.columns


# In[20]:

# Select columns to filter on
cols = [col for col in asthma_df.columns if 'HES_p' not in col and 'PC' not in col]


# In[21]:

asthma_df=asthma_df[cols].drop('Unnamed: 0',axis=1)


# In[22]:

asthma_df.columns


# In[23]:

asthma_df.std()


# In[29]:

asthma_df_mean = asthma_df.mean()


# In[30]:

asthma_df.mean()


# In[28]:

asthma_df_mean.max()


# In[31]:

asthma_df_mean.min()


# In[36]:

asthma_df_mean.sort_values()


# In[46]:

cor=pd.DataFrame.corr(asthma_df)


# In[64]:

correlated_variables=cor.unstack().sort_values(ascending=False)


# In[77]:

#print(correlated_variables.index[0],correlated_variables[0])
print(correlated_variables.head(50))


# In[78]:

print(cor.unstack().head(20))


# In[76]:

#print(cor.sort_values(by='sex',ascending=False).head())
print(cor.head())


# In[314]:

#select only 'QUANTITY' fields
asthma_df_quant=asthma_df.loc[:, asthma_df.columns.str.contains('QUANT')].fillna(0.0)
#select only fields with 10 or more unique values
for col in asthma_df_quant.columns:
    if len(asthma_df_quant[col].unique()) < 50 :
        asthma_df_quant.drop(col,inplace=True,axis=1)
#calculate correlation matrix, take absolute value
corr_matrix=asthma_df_quant.corr().abs()
#only keep upper half of the matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape)).astype(np.bool))
# filter to highly correlated 0.9-0.975 range
corr_var_df = upper.stack().where(lambda x: np.fabs(x) < 0.99).dropna().sort_values(ascending=False).reset_index()
# rename data frame varialbes
corr_var_df.columns = ['Variable1','Variable2','Correlation']
corr_var_df=corr_var_df.drop_duplicates(keep='first')


# In[256]:

corr_var_df.to_csv('uncorrelated pairs.csv')


# In[84]:

# from 0 to 3rd highest correlated variable
for i in xrange(2,4):
    boxplot=asthma_df.boxplot(column=[corr_var_df['Variable1'][i],corr_var_df['Variable2'][i]])
    scatterplot=asthma_df.plot.scatter(x=corr_var_df['Variable1'][i],y=corr_var_df['Variable2'][i])
    plt.show(boxplot)
    plt.show(scatterplot)


# In[103]:

i=2
boxplot=asthma_df.boxplot(column=[corr_var_df['Variable1'][i],corr_var_df['Variable2'][i]])
scatterplot=asthma_df.plot.scatter(x=corr_var_df['Variable1'][i],y=corr_var_df['Variable2'][i])
plt.show(boxplot)
plt.show(scatterplot)


# In[340]:

i=215
boxplot=asthma_df.boxplot(column=[corr_var_df['Variable1'][i],corr_var_df['Variable2'][i]])
scatterplot=asthma_df.plot.scatter(x=corr_var_df['Variable1'][i],y=corr_var_df['Variable2'][i])
plt.show(boxplot)
plt.show(scatterplot)


# In[312]:

boxplot=asthma_df.boxplot(column=['f_34_0_0_f_QUANT_Year_of_birth',corr_var_df['Variable2'][i]])
scatterplot=asthma_df.plot.scatter(x='f_34_0_0_f_QUANT_Year_of_birth',y='f_3063_0_f_QUANT_FEV1_maximumValue')
plt.show(boxplot)
plt.show(scatterplot)


# In[308]:

temp=[x for x in asthma_df.columns if 'Year_of_birth' in x]


# In[309]:

temp


# In[10]:

import pandas as pd
df = pd.read_csv('correlated pairs.csv')


# In[16]:

df.columns.values
df.Variable2.unique() == df.Variable1.unique()


# In[ ]:




# In[ ]:



