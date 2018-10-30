
# coding: utf-8

# In[ ]:

from pyspark.sql import SparkSession


# In[ ]:

print('hello')


# In[2]:

sparkSession = SparkSession.builder.appName("example-pyspark-read-and-write").getOrCreate()


# In[3]:

df_load = sparkSession.read.csv('hdfs:///data/unstructured/integrated/gene_e/ukbiobank/analysis-files/UKB9888.ALL.20180530.txt',header=True,inferSchema=True,sep='\t')


# In[4]:

#print df_load.count()


# In[4]:

df_asthma=df_load.filter(df_load.asthma==1.0)


# In[5]:

print df_asthma.count()


# In[6]:

print df_asthma.select('asthma').show(5)


# In[11]:

df_asthma.limit(5000).toPandas().to_csv("ukbb_asthma_sample_5k_new.csv",sep='\t')


# In[ ]:

#use .limit() to select rows


# In[8]:

df_asthma


# In[ ]:



