# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:28:05 2020

@author: Sejal Bhalla
"""
#reference: https://github.com/raj1603chdry/CSE3018-Content-Based-Image-and-Video-Retrieval-Lab/tree/master/WEEK4

import pandas as pd
import pickle
import os
import time
import termtables as tt
from statistics import mean
import math
from PIL import Image
import numpy as np
from IPython.display import display
import pandas as pd

def color_autocorr(I, dist):
  img = Image.open(I)
  img = img.resize((int(img.size[0]/4), int(img.size[1]/4)))
  #display(img)
  m = 256  #no. of colors
  #print('Quantizing image..')
  img_quant = img.quantize(m)
  width, height = img_quant.size
  #display(img_quant)
  color_correlogram = []

  pixels = np.array(list(img_quant.getdata())).reshape((width, height))

  d = len(dist)
  count = np.zeros((m, d))
  total = np.zeros((m, d))

  probs = np.zeros((d,m))

  #print('working it out..')
  i = 0
  while i < d:
    x = 0
    while x < width:
      y = 0
      while y < height:
        color = pixels[x][y]
        counts = neighbourhood(dist[i], x, y, color, pixels, width, height)
        pos_count = counts[0]
        total_count = counts[1]
        count.put((color*d)+i, count[color][i]+pos_count)
        total.put((color*d)+i, total[color][i]+total_count) 
        y += 1
      x += 1
    probs[i] = count[:,i]/total[:,i]
    i+=1

  for i in range(d):
    color_correlogram.insert(i, probs[i])

  return color_correlogram

  
def neighbourhood(d,x,y,color,pixels,X,Y):
  #d = distance
  #x, y = position of the pixel
  #color = color of the pixel
  #pixels = pixels of quantized image
  #X = actual width #Y = actual height
  x_neighb = np.zeros(8*d)
  y_neighb = np.zeros(8*d)

  ################ Y-coordinates ################
  #print('calculating y neighbourhood')
  #from 1 to d
  y_neighb[0] = y
  
  for k in range(1,d):
    y_neighb[k] = y - k
    

  #from d to 3d
  i = d
  while i < 3*d:
      y_neighb[i] = y-d
      i+=1
    
  #from 3d to 5d and 7*d to 8*d
  for k in range(3*d,8*d):
      if k >= 3*d and k < 5*d:
        y_neighb[k] = y - d + (k - 3*d)
      elif k >= 7*d and k < 8*d:
          y_neighb[k] = y + d - (k-7*d)
        
  #from 5d to 7d
  i = 5*d
  while i < 7*d:
      y_neighb[i] = y+d
      i+=1


  ################ X-coordinates ################
  #print('calculating x neighbourhood')
  #from 0 to d
  i = 0
  while i < d:
      x_neighb[i] = x-d
      i+=1

  #from d to 3d and 5*d to 7*d
  for k in range(d,7*d):
      if k >= d and k < 3*d:
          x_neighb[k] = x - d + (k-d)
      elif k >= 5*d and k < 7*d:
          x_neighb[k] = x + d - (k - 5*d)

  #from 3d to 5d
  i = 3*d
  while i < 5*d:
      x_neighb[i] = x+d
      i+=1
  
  #from 7d to end
  i = 7*d
  while i < 8*d:
      x_neighb[i] = x-d
      i+=1

  ###################### Determining the neighbourhood ###############################
  pos_count=0   
  total_count=0
  for i in range(8*d):
    if x_neighb[i] < 0 and x_neighb[i] > X:
        continue
    if y_neighb[i] < 0 and y_neighb[i] > Y:
        continue
    neighb_color = pixels[int(x_neighb[i])][int(y_neighb[i])]
    total_count = total_count+1
    if neighb_color != color:
        continue
    else:
        pos_count = pos_count + 1

    return pos_count, total_count

#
#""" Calculate Features """
#features = {}
#distances = [1,3,5]
#
#directory = os.fsencode(str('images/'))
#i = 0
#for file in os.listdir(directory):
#  filename = os.fsdecode(file)
#  if filename.endswith(".jpg"):
#    print("calculating features of "+filename+", "+str(i)) 
#    features[str(filename)] = np.array(color_autocorr(str('images/'+filename), distances)).flatten()
#  i+=1
  
  

def similarity(query_corr, sample_corr, m):
  #return similarity between correlograms of 2 images
  return ((1/m)*(sum([(abs(a-b)/(1+a+b)) for a, b in zip(query_corr, sample_corr)])))

def euclidean_dist(query_corr, sample_corr, m):
    return math.sqrt(sum([(a-b)*(a-b) for a, b in zip(query_corr, sample_corr)]))

def man_dist(query_corr, sample_corr, m):
    return sum([abs(a-b) for a, b in zip(query_corr, sample_corr)])

def sort_score(l):
    l.sort(key = lambda x: x[1], reverse = True)
    return l


features = pickle.load(open('features256.pkl', 'rb'))

#m = 128
#d = 3
#cols = ['feature'+str(i+1) for i in range(m*d)]
#features = pd.DataFrame.from_dict(features, orient='index', columns=cols)

query_dir = os.fsencode(str('train/query/'))

good_percent = []
junk_percent = []
ok_percent = []

precision = []
recall = []
f1 = []

retrieval_time = []

for f in os.listdir(query_dir):
    start = time.time()  #start time of retrieval
    file_name = os.fsdecode(f)
    file = open('train/query/'+file_name)
    line = file.readline()
    file.close()
    name = line.split(" ")[0][5:]
    query_corr = features.loc[name+'.jpg'].values.tolist()
    scores = [(img, similarity(query_corr, features.loc[img].values.tolist(), 256)) for img in features.index if img != str(name+'.jpg')]
    scores = sort_score(scores)
    
    good_file = open('train/ground_truth/'+file_name[:-9]+'good.txt')
    junk_file = open('train/ground_truth/'+file_name[:-9]+'junk.txt')
    ok_file = open('train/ground_truth/'+file_name[:-9]+'ok.txt')
    
    good_images = []
    junk_images = []
    ok_images = []
    
    for line in good_file:
        good_images.append(line.rstrip('\n'))
        
    for line in junk_file:
        junk_images.append(line.rstrip('\n'))
        
    for line in ok_file:
        ok_images.append(line.rstrip('\n'))
    
    n = len(good_images) + len(junk_images) + len(ok_images) #top n images to be retrieved
    top_n = [a[0] for a in scores][:n]
    
    end = time.time()  #end time of retrieval
    
    good_retrieved = []
    junk_retrieved = []
    ok_retrieved = []
    
    good_len = 0
    junk_len = 0
    ok_len = 0
    
    for a in top_n:
        if a[:-4] in good_images:
            good_retrieved.append(a)
            good_len+=1
        elif a[:-4] in junk_images:
            junk_retrieved.append(a)
            junk_len+=1
        elif a[:-4] in ok_images:
            ok_retrieved.append(a)
            ok_len+=1
    
    good_percent.append(good_len*100/len(good_images))
    junk_percent.append(junk_len*100/len(junk_images))
    ok_percent.append(ok_len*100/len(ok_images))
    
    precision.append((good_len+junk_len+ok_len)/n)
    recall.append((good_len+junk_len+ok_len)/n)
    
    f1.append(precision[-1])  #f1 = precison = recall by construction
     
    retrieval_time.append(end-start)
    
    
result = tt.to_string([["Maximum Precision", max(precision)], ["Minimum Precision", min(precision)],
                        ["Average Precision", mean(precision)], ["Maximum Recall", max(recall)],
                        ["Minimum Recall", min(recall)], ["Average Recall", mean(recall)],
                        ["Maximum F1 Score", max(f1)], ["Minimum F1 Score", min(f1)],
                        ["Average F1 Score", mean(f1)], ["Average Retrieval Time", mean(retrieval_time)],
                        ["Average Good Retrieved", mean(good_percent)], ["Average Junk Retrieved", mean(junk_percent)],
                        ["Average Ok Retrieved", mean(ok_percent)]],
    header=["Metric", "Value"],
    style=tt.styles.ascii_thin_double,)

print(result)