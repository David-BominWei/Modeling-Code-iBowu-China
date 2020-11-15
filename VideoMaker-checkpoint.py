#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import NMF


# In[2]:


image = cv2.imread('fig0.png')


# In[3]:


fps = 20.0
size = (int(image.shape[1]),int(image.shape[0]))


# In[4]:


fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('out.avi',fourcc,fps,size)


# In[5]:


N = 500
for i in range(N+1):
    filename = 'fig'+str(i)+'.png'
    im = cv2.imread(filename)
    
    out.write(im)
    
cv2.destroyAllWindows()


# In[ ]:




