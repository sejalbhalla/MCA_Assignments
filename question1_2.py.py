#Reference: https://github.com/n-k-chilagani/cv-assignment1/tree/master/Scale%20Space%20Blob%20Detection

import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def LoG(sigma):
    #window size 
    idx = np.ceil(sigma*6)//2
    y,x = np.ogrid[-idx:idx + 1, -idx:idx+1]
    y_filt = np.exp(-(y*y/(2.*np.power(sigma, 2))))
    x_filt = np.exp(-(x*x/(2.*np.power(sigma, 2))))
    final_filter = (-(2*np.power(sigma,2) + (np.square(x) + np.square(y))) *  (x_filt*y_filt) * (1/(2*np.pi*np.power(sigma, 4))))
    return final_filter

def LoG_convolve(img):
    log_images = [] #to store responses
    i = 0
    while i < 9:
        sigma_1 = sigma*(k ** i) #sigma 
        filter_log = LoG(sigma_1) #filter generation
        image = cv2.filter2D(img,-1,filter_log) # convolving image
        pad = (1,1)
        image = np.pad(image,(pad, pad),'constant') #padding 
        image = np.power(image, 2) # squaring the response
        log_images.insert(i, image)
        i+=1
    return np.array([i for i in log_images])

#print(log_image_np.shape)
    
def detect_blob(log_image_np):
    co_ordinates = [] #to store co ordinates
    (h,w) = img.shape
    index_coordinates = 0
    for i in range(1,h,1):
        for j in range(1,w,1):
            idx_i = i-1
            idx_j = j-1
            slice_img = log_image_np[:, idx_i:idx_i+3, idx_j:idx_j+3] #9*3*3 slice
            result = np.amax(slice_img) #finding maximum
            shape = slice_img.shape
            #print(shape)
            z_size = shape[0]
            x_size = shape[1]
            y_size = shape[2]
            if result >= 0.03: #threshold
                max_idx = slice_img.argmax()
                y = max_idx % y_size
                x = ((max_idx - y)/y_size) % x_size
                z = (max_idx - y - (x*y_size))/(x_size*y_size)
                #z,x,_ = np.unravel_index(slice_img.argmax(),shape)
                #_,_,y = np.unravel_index(slice_img.argmax(),shape)
                co_ordinates.insert(index_coordinates, (i+x-1, j+y-1, np.power(k, z)*sigma)) #finding co-ordinates
                index_coordinates += 1
    return co_ordinates


query_dir = os.fsencode(str('train/query/'))
query = []
blobs = {}

k = 1.414
sigma = 1.0
for i in range(5):
    f = os.listdir(query_dir)[i]
    #start = time.time()  #start time of retrieval
    file_name = os.fsdecode(f)
    file = open('train/query/'+file_name)
    line = file.readline()
    file.close()
    name = line.split(" ")[0][5:]
    query.append(name+'.jpg')
    
    img = cv2.imread("images/"+query[-1], 0) #gray scale conversion

    img = img/255.0  #image normalization

    log_image_np = LoG_convolve(img)
    co_ordinates = list(set(detect_blob(log_image_np)))
    
    blobs[query[-1]] = co_ordinates

#    fig, ax = plt.subplots()
#    height, width = img.shape
#    count = 0
#    
#    ax.imshow(img, interpolation='nearest',cmap="gray")
#    for blob in co_ordinates:
#        y,x,r = blob
#        c = plt.Circle((x, y), r*1.414, color='red', linewidth=1.5, fill=False)
#        ax.add_patch(c)
#    ax.plot()    
#    plt.show() 