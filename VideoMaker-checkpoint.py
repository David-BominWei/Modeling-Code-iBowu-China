'''
!/usr/bin/env python
coding: utf-8
'''

import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import NMF



image = cv2.imread('fig0.png')


fps = 20.0
size = (int(image.shape[1]),int(image.shape[0]))

fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('out.avi',fourcc,fps,size)



N = 500
for i in range(N+1):
    filename = 'fig'+str(i)+'.png'
    im = cv2.imread(filename)
    
    out.write(im)
    
cv2.destroyAllWindows()





