#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('train_binary.csv')
df.head()


# In[3]:


df.shape


# In[4]:


import cv2

train_paths = "train_images/"

def load_images(image_paths,df):
    loadedImages = []
    
    for img in df.imagesID:
        image = cv2.imread(train_paths+img)
        loadedImages.append(image)
    return loadedImages

train_images = load_images(train_paths,df)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(train_images[100].shape)
plt.imshow(train_images[0])


# In[6]:


def resize_images(images):
    resizedimages = []  
    for img in images:  
        img = cv2.resize(img,(256,img.shape[0]))
        resizedimages.append(img)
    return resizedimages

train_images_r = resize_images(train_images)


# In[7]:


print(train_images_r[0].shape)
plt.imshow(train_images_r[0])


# In[8]:



def features_to_np_array(images):
    imagenp = np.empty(shape = (len(images),images[0].shape[0],images[0].shape[1],images[0].shape[2]), dtype='uint8')
    idx = 0
    for img in images:
        imagenp[idx,:,:,:] = img[:,:,:]
        idx = idx+1
    imagenp = imagenp.reshape((imagenp.shape[0],imagenp.shape[1]*imagenp.shape[2]*imagenp.shape[3]))
    return imagenp
    
    
train_images = features_to_np_array(train_images_r)


# In[9]:


train_images.shape


# In[10]:


y = pd.get_dummies(df.label,columns='label')
y = np.array(y)


# In[11]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_images,y,test_size = 0.2,random_state=42, stratify=y)
X_train.shape


# In[12]:


X_train_r = X_train.reshape((10054,256,256,3))
X_train_r.shape


# In[13]:


y_test.shape


# In[14]:


X_test_r = X_test.reshape((2514,256,256,3))
X_test_r.shape


# In[15]:


import tensorflow as tf
import random as rn

# Set up your models here
# Setting the seed for numpy-generated random numbers
np.random.seed(37)

# Setting the seed for python random numbers
rn.seed(1254)

# Setting the graph-level random seed.
tf.random.set_seed(89)

from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D

model = Sequential()
#model_28.add(Dense(128,input_dim = 784,activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), input_shape= (256,256,3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[16]:


model.fit(X_train_r,y_train,epochs=10,batch_size=128)


# In[18]:


model.evaluate(X_test_r, y_test)


# In[ ]:




