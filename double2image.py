import pandas as pd
import numpy as np

def to_image(m):
    '''
    convert an array of doubles to 0-255 image format
    input:
        m : numpy 2d array 
    output:
        img : numpy 3d array of 0-255 ints
    '''
    tab = pd.read_csv('./CoolWarmFloat257.csv')
    m_min = np.min(m)
    m_max = np.max(m)
    if m_max != m_min:
        m = np.floor((m - m_min)/(m_max - m_min) * 255) + 1
    else:
        m = np.floor((m - m_min)) + 1
    img_size = np.shape(m) + (3,)
    img = np.zeros(img_size)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            img[i, j, ] =  (tab.R[m[i, j]], tab.G[m[i, j]], tab.B[m[i, j]])
    return img
                
