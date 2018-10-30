
# coding: utf-8

# In[33]:

import pandas as pd 
import numpy as np
get_ipython().magic(u'matplotlib inline')
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(12,6)


# In[34]:

#Read in 5K sample
asthma_df=pd.read_csv("ukbb_asthma_sample_5k_new.csv",sep='\t')


# ## Read in the top uncorrelated pairs File, and filter the data

# In[35]:

top_uncorrelated_df=pd.read_csv('uncorrelated_pairs_final.csv',header=None)
top_uncorrelated_df=top_uncorrelated_df.rename(columns={0:'Uncorrelated Rank',1:'Feature 1',2:'Feature 2',3:'Correlation'})


# In[53]:

top_uncorrelated_df.head()


# In[36]:

# Select Columns
cols=np.append(top_uncorrelated_df['Feature 1'].unique(),top_uncorrelated_df['Feature 2'].unique())
cols=np.unique(cols)
asthma_df_uncorrelated=asthma_df[cols]


# In[39]:

asthma_df_uncorrelated_info = asthma_df_uncorrelated.describe()
print(asthma_df_uncorrelated_info.T)
asthma_df_uncorrelated_info.T.to_csv('uncorrelated pairs info.csv')


# ## Task #2) Get the Mean and Standard Deviations for all the uncorrelated variables. 
# 
# Hints:
# * The describe function is your friend
# * the asthma_df_uncorrelated data frame is already filtered for the uncorrelated variables
# * Print this table to the screen, and save it as a CSV table with the to_csv() function
# 

# # Task #3) Table of most correlated variables along with correlation values
# 
# This task is actually complete with your manually created CSV files of the most important correlated and uncorrelated variables

# ## Task #4) Box plots and scatter plots of final correlated variable list
# 
# Below, I plot just one of the variables and save it as a file. Your task: make a for loop that looks through all the variable pairs, make the plots, and save the output. 

# In[27]:

for i in range(0,len(top_uncorrelated_df)):
    #print(i)
    boxplot=asthma_df_uncorrelated.boxplot(column=[top_uncorrelated_df['Feature 1'][i],top_uncorrelated_df['Feature 2'][i]],rot=90)
    scatterplot=asthma_df_uncorrelated.plot.scatter(x=top_uncorrelated_df['Feature 1'][i],y=top_uncorrelated_df['Feature 2'][i])
    plt.show(boxplot)
    boxplot.get_figure().savefig('boxplot_uncorrelated_'+str(top_uncorrelated_df['Uncorrelated Rank'][i])+'.png',bbox_inches='tight')
    plt.show(scatterplot)
    scatterplot.get_figure().savefig('scatterplot_uncorrelated_'+str(top_uncorrelated_df['Uncorrelated Rank'][i])+'.png',bbox_inches='tight')



#for i in xrange(2,4):
#    boxplot=asthma_df.boxplot(column=[corr_var_df['Variable1'][i],corr_var_df['Variable2'][i]])
#    scatterplot=asthma_df.plot.scatter(x=corr_var_df['Variable1'][i],y=corr_var_df['Variable2'][i])
#    plt.show(boxplot)
#    plt.show(scatterplot)


# In[23]:

#for i in asthma_df_uncorrelated
i=0
boxplot=asthma_df_uncorrelated.boxplot(column=[top_uncorrelated_df['Feature 1'][i],top_uncorrelated_df['Feature 2'][i]],rot=90)
scatterplot=asthma_df_uncorrelated.plot.scatter(x=top_uncorrelated_df['Feature 1'][i],y=top_uncorrelated_df['Feature 2'][i])
plt.show(boxplot)
boxplot.get_figure().savefig('boxplot_uncorrelated_'+str(top_uncorrelated_df['Uncorrelated Rank'][i])+'.png',bbox_inches='tight')
plt.show(scatterplot)
scatterplot.get_figure().savefig('scatterplot_uncorrelated_'+str(top_uncorrelated_df['Uncorrelated Rank'][i])+'.png',bbox_inches='tight')



# ## Repeat the above, but for the top Correlated Variables

# In[30]:

top_correlated_df=pd.read_csv('correlated_pairs_final_.csv',header=None)
top_correlated_df=top_correlated_df.rename(columns={0:'Correlated Rank',1:'Feature 1',2:'Feature 2',3:'Correlation'})


# In[14]:

top_correlated_df.head()


# In[31]:

# Select Columns
cols=np.append(top_correlated_df['Feature 1'].unique(),top_correlated_df['Feature 2'].unique())
cols=np.unique(cols)
asthma_df_correlated=asthma_df[cols]


# In[40]:

asthma_df_correlated_info = asthma_df_correlated.describe()
print(asthma_df_correlated_info.T)
asthma_df_correlated_info.T.to_csv('correlated pairs info.csv')


# In[32]:

for i in range(0,len(top_correlated_df)):
    #print(i)
    boxplot=asthma_df_correlated.boxplot(column=[top_correlated_df['Feature 1'][i],top_correlated_df['Feature 2'][i]],rot=90)
    scatterplot=asthma_df_correlated.plot.scatter(x=top_correlated_df['Feature 1'][i],y=top_correlated_df['Feature 2'][i])
    plt.show(boxplot)
    boxplot.get_figure().savefig('boxplot_correlated_'+str(top_correlated_df['Correlated Rank'][i])+'.png',bbox_inches='tight')
    plt.show(scatterplot)
    scatterplot.get_figure().savefig('scatterplot_correlated_'+str(top_correlated_df['Correlated Rank'][i])+'.png',bbox_inches='tight')


# In[ ]:



