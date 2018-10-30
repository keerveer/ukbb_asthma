
# coding: utf-8

# In[1]:

import pandas as pd


# In[67]:

import numpy as np
get_ipython().magic(u'matplotlib inline')
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(12,6)


# In[245]:

#read in data
asthma_df=pd.read_csv("ukbb_asthma_sample_5k.csv",sep='\t')


# In[246]:

asthma_df.columns


# In[130]:

#test_cols = [col for col in data_df.columns if 'HES_p' in col]


# In[131]:

#data_df[test_cols]['HES_p_A02_BIN_Other_salmonella_infections']


# In[ ]:

#test_df[test_cols].astype(bool).sum(axis=0).sum(axis=0)


# In[ ]:

count=0
total=0

for col in test_df[test_cols].columns:
    count+=test_df[col].sum()
    total+=len(test_df[col])

print count
print float(count)/float(total)


# In[272]:

#select only 'QUANTITY' fields
asthma_df_quant=asthma_df.iloc[:,asthma_df.columns.str.contains('QUANT')].fillna(0.0)


# In[274]:

#select only fields with 10 or more unique values
for col in asthma_df_quant.columns:
    if len(asthma_df_quant[col].unique()) < 50 :
        print col
        asthma_df_quant.drop(col,inplace=True,axis=1)


# In[273]:

asthma_df_quant.head()


# In[266]:

#calculate correlation matrix, take absolute value
corr_matrix=asthma_df_quant.corr().abs()
#only keep upper half of the matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape)).astype(np.bool))
# filter to highly correlated 0.9-0.975 range
corr_var_df = upper.stack().where(lambda x: np.fabs(x) < 0.99).where(lambda x: np.fabs(x) > 0.975).dropna().sort_values(ascending=False).reset_index()
# rename data frame varialbes
corr_var_df.columns = ['Variable1','Variable2','Correlation']
corr_var_df=corr_var_df.drop_duplicates(keep='first')


# In[268]:

corr_var_df.shape


# In[259]:

for i in xrange(0,3):
    boxplot=asthma_df.boxplot(column=[corr_var_df['Variable1'][i],corr_var_df['Variable2'][i]])
    scatterplot=data_df.plot.scatter(x=corr_var_df['Variable1'][i],y=corr_var_df['Variable2'][i])
    plt.show(boxplot)
    plt.show(scatterplot)


# In[ ]:



