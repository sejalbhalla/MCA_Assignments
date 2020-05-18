#Reference: https://medium.com/@lerner98/implementing-sift-in-python-36c619df7945

from scipy.ndimage.filters import convolve
import numpy as np
from numpy import linalg as LA
from PIL import Image
import os
import time
import matplotlib.pyplot as plt

def generate_octave(init_level, s, sigma): 
    #octave = set of images where the blur of the last image = 2*the blur of the 1st image
    #s = no. of images per octave
    #sigma = 2^(1/s)
  octave = []
  octave.append(init_level)
  k = 2**(1/s) 
  kernel = gaussian_filter(k * sigma) 
  for i in range(s+2): 
    octave.append(convolve(octave[-1], kernel))
  return octave

def gaussian_filter(sigma):
  temp_size = np.ceil(3*sigma)
  size = (2*temp_size)+1 
  start = (-size//2) + 1
  end = (size//2) + 1
  x, y = np.mgrid[start:end, start:end] 
  g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
  filt = g/g.sum()
  return filt

def generate_DoG_octave(gaussian_octave): 
  octave = [] 
  for i in range(1, len(gaussian_octave)):   
      dog_octave = gaussian_octave[i] - gaussian_octave[i-1]
      octave.append(dog_octave) 
  temp = [o[:,:,np.newaxis] for o in octave]
  return np.concatenate(temp, axis=2) 


def generate_DoG_pyramid(img, num_octave, s, sigma): 
  pyr = [] 
  for x in range(num_octave): 
    octave = generate_octave(img, s, sigma)  #part of the gaussian pyramid
    pyr.append(generate_DoG_octave(octave)) 
    img = octave[-3][::2, ::2] 
  return pyr


def get_candidate_keypoints(D, w=16): 
    #w = side length of the patches used when creating local descriptors
  candidates = []
  #top and bottom levels = 0
  D[:,:,0] = 0  
  D[:,:,-1] = D[:,:,0]
  start = w//2 + 1
  poi = 13  #27/2 = 13, point of interest
  end_outer = D.shape[0]-(w//2)-1
  end_inner = D.shape[1]-(w//2)-1
  for i in range(start, end_outer): 
    for j in range(start, end_inner): 
      for k in range(1, D.shape[2]-1): 
        patch = D[i-1:i+2, j-1:j+2, k-1:k+2] 
        if np.argmax(patch) == poi or np.argmin(patch) == poi:  #to detect extrema
            extrema = [i, j, k]
            candidates.append(extrema) 
  return candidates

def localize_keypoint(D, x, y, s): 
  J = Jacobian(D, x, y, s)
  HD = Hessian(D, x, y, s)
  offset = -LA.inv(HD).dot(J)
  return offset, J, HD[:2,:2], x, y, s

def Jacobian(D, x, y, s):
    dx = (D[y,x+1,s]-D[y,x-1,s])/2. 
    dy = (D[y+1,x,s]-D[y-1,x,s])/2. 
    ds = (D[y,x,s+1]-D[y,x,s-1])/2. 
    J = np.array([dx, dy, ds])
    return J

def Hessian(D, x, y, s):
    dxx = D[y,x+1,s]-2*D[y,x,s]+D[y,x-1,s] 
    dxy = ((D[y+1,x+1,s]-D[y+1,x-1,s]) - (D[y-1,x+1,s]-D[y-1,x-1,s]))/4.0 
    dxs = ((D[y,x+1,s+1]-D[y,x-1,s+1]) - (D[y,x+1,s-1]-D[y,x-1,s-1]))/4.0
    dyy = D[y+1,x,s]-2*D[y,x,s]+D[y-1,x,s]
    dys = ((D[y+1,x,s+1]-D[y-1,x,s+1]) - (D[y+1,x,s-1]-D[y-1,x,s-1]))/4.0
    dss = D[y,x,s+1]-2*D[y,x,s]+D[y,x,s-1] 
    HD = np.array([ [dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]) 
    return HD

def find_keypoints_for_DoG_octave(D, R_th, t_c, w): 
  candidates = get_candidate_keypoints(D, w)
  keypoints = [] 
  for i, candidate in enumerate(candidates): 
    y, x, s = candidate[0], candidate[1], candidate[2] 
    offset, J, H, x, y, s = localize_keypoint(D, x, y, s) 
    contrast = D[y,x,s] + .5*J.dot(offset) 
    if abs(contrast) < t_c: continue 
    w, v = LA.eig(H) 
    r = w[1]/w[0] 
    R = (r+1)**2 / r 
    if R > R_th: continue 
    kp = np.array([x, y, s]) + offset
    keypoints.append(kp)
  return np.array(keypoints)

def get_keypoints(DoG_pyr, R_th, t_c, w): 
  kps = [] 
  for D in DoG_pyr: 
    kps.append(find_keypoints_for_DoG_octave(D, R_th, t_c, w)) 
  return kps


query_dir = os.fsencode(str('train/query/'))
query = []

s=3
num_octave=4
s0=1.3
sigma=1.6
r_th=10
t_c=0.03  #from paper
w=16
R_th = (r_th+1)**2 / r_th  #from paper
key_points = {}

for i in range(5):
    f = os.listdir(query_dir)[i]
    start = time.time()  #start time of retrieval
    file_name = os.fsdecode(f)
    file = open('train/query/'+file_name)
    line = file.readline()
    file.close()
    name = line.split(" ")[0][5:]
    query.append(name+'.jpg')
    
    img = Image.open("images/"+query[-1])
    img = convolve(np.array(img.convert('L')), gaussian_filter(1.3))
    
    #gaussian_pyr = generate_gaussian_pyramid(img, num_octave, s, sigma) 
    DoG_pyr = generate_DoG_pyramid(img, num_octave, s, sigma) 
    kp_pyr = get_keypoints(DoG_pyr, R_th, t_c, w) 
    
    key_points[query[-1]] = kp_pyr
    
#    _, ax = plt.subplots(1, num_octave)
#    
#    for i in range(num_octave):
#        ax[i].imshow(img)
#        scaled_kps = kp_pyr[i] * (2**i)
#        ax[i].scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=2.5)
#        
#    plt.show()
     

    

