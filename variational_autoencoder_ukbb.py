
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics
import pandas as pd
#import tensorflow as tf
import keras


# In[3]:

import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().magic(u'matplotlib inline')
#import pydot
#import graphviz
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[4]:

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import manifold


# In[5]:

asthma_df=pd.read_csv("ukbb_asthma_sample_5k.csv",sep='\t')


# In[7]:

# Select columns to filter on
cols = [col for col in asthma_df.columns if 'HES_p' not in col and 'PC' not in col]
asthma_df=asthma_df[cols].drop('Unnamed: 0',axis=1)


# In[8]:

#select only 'QUANTITY' fields
asthma_df_quant=asthma_df.loc[:, asthma_df.columns.str.contains('QUANT|age|BMI')].fillna(0.0)
#select only fields with 10 or more unique values
for col in asthma_df_quant.columns:
    if len(asthma_df_quant[col].unique()) < 20 :
        asthma_df_quant.drop(col,inplace=True,axis=1)

        
#select only 'QUANTITY' and "CATEGORY" fields
asthma_df_quant_cat=asthma_df.loc[:, asthma_df.columns.str.contains('QUANT|age|BMI|CAT|sex')].fillna(0.0)
asthma_df_quant_cat=asthma_df_quant_cat.drop(columns=['f_22182_0_0_f_CAT_HLA_imputation_values_and_quality'])


# In[17]:

#asthma_df_quant_cat['f_40019_0_p_D44_CAT_Neoplasm_of_uncertain_or_unknown_behaviour_of_endocrine_glands'].value_counts()


# In[18]:

#asthma_df_quant_cat.loc[:,asthma_df_quant_cat.columns.str.contains('CAT')]


# In[20]:

asthma_df_quant.head()


# In[73]:

#data=asthma_df_quant.transpose().values
asthma_df_quant_scaled=asthma_df_quant
scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
asthma_df_quant_scaled[asthma_df_quant_scaled.columns]=scaler.fit_transform(asthma_df_quant[asthma_df_quant.columns])


# In[86]:

plt.scatter(asthma_df['age'].values,asthma_df_quant_scaled['age'].values)


# In[89]:

data_minmax=asthma_df_quant_scaled.values


# In[91]:

print data_minmax.shape


# In[92]:

batch_size = 50
original_dim = data.shape[1]
#latent_dim = 2
latent_dim=12
#intermediate_dim = 500
epochs = 200
epsilon_std = 1.0
learning_rate=0.0005
momentum=0.9


# In[93]:

def sampling(args):
    
    import tensorflow as tf
    epsilon_std=1.0
    
    z_mean, z_log_var = args
    #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                             # stddev=epsilon_std)
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - K.exp(z_log_var_encoded), axis=-1)
        return K.mean(xent_loss + kl_loss)


# In[94]:

#Encoder
x = Input(shape=(original_dim, ))

# Input layer is compressed into a mean and log variance vector of size `latent_dim`
# Each layer is initialized with glorot uniform weights and each step (dense connections, batch norm,
# and relu activation) are funneled separately
# Each vector of length `latent_dim` are connected to the input tensor
# Use dropout to help with training and overfitting
# Use batch normalization for regularization

dropout_frac=0.1
dropout_mean=Dropout(dropout_frac, input_shape=(original_dim,))(x)
dropout_var=Dropout(dropout_frac, input_shape=(original_dim,))(x)

z_mean_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(dropout_mean)
z_mean_dense_batchnorm = keras.layers.BatchNormalization(momentum=momentum)(z_mean_dense_linear)
z_mean_encoded = keras.layers.Activation('relu')(z_mean_dense_batchnorm)

z_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(dropout_var)
z_log_var_dense_batchnorm = keras.layers.BatchNormalization(momentum=momentum)(z_log_var_dense_linear)
z_log_var_encoded = keras.layers.Activation('relu')(z_log_var_dense_batchnorm)

# return the encoded and randomly sampled z vector
# Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean_encoded, z_log_var_encoded])


# In[95]:

#Decoder
decoder = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
expression_reco = decoder(z)


# In[96]:

np.random.seed(123)


# In[97]:

adam = keras.optimizers.Adam(lr=learning_rate)

#vae_layer = CustomVariationalLayer()([x, expression_reconstruct])
#vae = Model(x, vae_layer)
vae = Model(x, expression_reco)
#RMS prop seems to be the most stable optimizer for this config
vae.compile(optimizer="rmsprop", loss=vae_loss)
#vae.compile(optimizer=adam, loss=vae_loss)

print vae.summary()


# In[99]:

import time
start = time.clock()
data_split=0.2
data_train, data_test = train_test_split(data_minmax, test_size=data_split)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_AN_Variational', histogram_freq=0, write_graph=True, write_images=True)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks_list = [tbCallBack,earlystop]


# In[100]:

hist1=vae.fit(data_train,data_train,
        #shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks_list)


# In[103]:

#Visualize training performance
history_df = pd.DataFrame(hist1.history)
ax = history_df[1:].plot()
ax.set_xlabel('Epochs')
ax.set_ylabel('VAE Loss')
fig = ax.get_figure()
#plt.yscale('log')


# In[104]:

trained = time.clock()
print "Time to train: ",trained-start


# In[105]:

# encoder, from inputs to latent space
encoder = Model(x, z_mean_encoded)
#encoder = Model(x, z)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
decoded_latent=decoder(decoder_input)
generator=Model(decoder_input,decoded_latent)


# In[106]:

#data_predict=vae.predict(data_test,batch_size=batch_size)
data_predict=vae.predict_on_batch(data_test)
diffs=np.subtract(data_predict,data_test).flatten()


# In[113]:

plt.hist(diffs, bins='auto')  # plt.hist passes it's arguments to np.histogram

plt.title("Predicted phenotype minus real phenotype (normalized)")
plt.xlim(-0.3,0.3)
plt.show()
print np.mean(diffs),np.std(diffs)


# In[115]:

encoded_predict=encoder.predict(data_test)


# In[140]:

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=50,
                     learning_rate=10, n_iter=1000)
#tsne = manifold.TSNE(n_components=2, init='pca')

tsne_out = tsne.fit_transform(encoded_predict)



# In[141]:

plt.scatter(tsne_out[:,0],tsne_out[:,1])

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Autoencoder QUANTITY FEATURES ONLY tSNE with two components')
plt.show()


# In[122]:

from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca_out=pca.fit_transform(encoded_predict)

plt.scatter(pca_out[:,0],pca_out[:,1])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Autoencoder QUANTITY FEATURES ONLY')
plt.show()


# In[ ]:



