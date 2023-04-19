# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 00:17:06 2022

@author: nicof
"""

import numpy as np
from PIL import Image
import math
from os import listdir
from os.path import isfile, join

Haar_2x2 = np.array([[1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]])
Haar_2x2_inv = np.transpose(Haar_2x2)

Haar_4x4 = np.array([[1/2, 1/2, 1/math.sqrt(2), 0], [1/2, 1/2, -1/math.sqrt(2), 0], [1/2, -1/2, 0, 1/math.sqrt(2)], [1/2, -1/2, 0, -1/math.sqrt(2)]])
Haar_4x4_inv = np.transpose(Haar_4x4)

images = [f for f in listdir('data/') if isfile(join('data/', f))]

def adjust_dimension(im_name, Haar_dim):
    rgb_img = Image.open('data/'+im_name).convert('RGB')
    img_arr = np.array(rgb_img)
    
    height = img_arr.shape[0]
    width = img_arr.shape[1]
    
    if (height%Haar_dim != 0) or (width%Haar_dim != 0): 
        print('resizing')
        height = height - height%Haar_dim
        width = width - width%Haar_dim
        rgb_img = rgb_img.resize((width, height))
        rgb_img.save('data/' + im_name)
    return 0

    

def get_compressed(im_name, Haar_dim, epsilon = 5):
    rgb_img = Image.open('data/'+im_name).convert('RGB')
    # get rid of transparancy component
    gray_img = rgb_img.convert('L')
    gray_img.save('output/gray_' + im_name)
    gray_img_arr = np.array(gray_img)
    
    height = gray_img_arr.shape[0]
    width = gray_img_arr.shape[1]
        
    # split the image into blocks of m x n pixels
    m = Haar_dim
    n = Haar_dim
    
    gray_tiles = [gray_img_arr[x:x+m,y:y+n] for x in range(0, height, m) for y in range(0, width, n)]
    gray_tiles_arr = np.array(gray_tiles)
    #gray_tiles_arr_reshape = gray_tiles_arr.transpose(1, 0, 2).reshape(1150, 2048)
    
    # COMPRESSION
    # multiply by Haar_4x4^-1 = Haar_4x4^T
    encoded_gray_arr = np.matmul(Haar_4x4_inv, gray_tiles_arr)
    # transpose each tile
    encoded_gray_arr = encoded_gray_arr.transpose(0, 2, 1)
    # multiply again by Haar_4x4^T
    encoded_gray_arr = np.matmul(Haar_4x4_inv, encoded_gray_arr)
    # transpose each tile again
    encoded_gray_arr = encoded_gray_arr.transpose(0, 2, 1)
    # values below epsilon get set to 0
    compressed_gray_arr = np.copy(encoded_gray_arr)
    compressed_gray_arr[compressed_gray_arr < epsilon] = 0
    
    ratio = np.count_nonzero(compressed_gray_arr==0)/np.count_nonzero(encoded_gray_arr==0)
    
    # RECOVERING IMAGE FROM BLOCKS
    temp_list = (np.empty(shape=(height//m, width//n))).tolist()
    index = 0
    for i in range(height//m):
        for j in range(width//n):
            temp_list[i][j] = compressed_gray_arr[index]
            index += 1
    
    temp_arr = np.array(temp_list).transpose(0, 2, 1, 3).reshape(height, width)
    temp_img = Image.fromarray(temp_arr)
    temp_img = temp_img.convert('L') #temp_img.convert("L", palette=Image.ADAPTIVE, colors=8) # to keep the bit depth = 8
    temp_img.save('output/transformed_' + im_name)
    
    
    # get back the compressed original
    compressed_gray_arr = np.matmul(Haar_4x4, compressed_gray_arr)
    compressed_gray_arr = np.matmul(compressed_gray_arr, Haar_4x4_inv)
    compressed_gray_arr = np.int8(compressed_gray_arr)
    
    
    # RECOVERING IMAGE FROM BLOCKS
    compressed_list = (np.empty(shape=(height//m, width//n))).tolist()
    index = 0
    for i in range(height//m):
        for j in range(width//n):
            compressed_list[i][j] = compressed_gray_arr[index]
            index += 1
    
    compressed_arr = np.array(compressed_list).transpose(0, 2, 1, 3).reshape(height, width)
    compressed_img = Image.fromarray(compressed_arr)
    compressed_img = compressed_img.convert("L", palette=Image.ADAPTIVE, colors=8) # to keep the bit depth = 8
    compressed_img.save('output/compressed_' + im_name)
    
    error = np.subtract(compressed_arr, gray_img_arr)
    #print(error)
    sq_error = np.square(error)/(height*width)
    sq_error[sq_error < 0] = 0 # to prevent rounding errors that lead to small, yet negative values
    #print(sq_error)
    MSE = np.sum(sq_error)
    #print(MSE)
    RMSE = math.sqrt(MSE)
    #print(RMSE)
    
    
    return ratio, RMSE

ratios = []
RMSEs = []

for im in images:
    print(im)
    adjust_dimension(im, 5)
    r, rmse = get_compressed(im, 4, 5)
    ratios.append(r)
    RMSEs.append(rmse)

print(ratios)
print(RMSEs)
print(sum(ratios)/len(ratios))
print(sum(RMSEs)/len(RMSEs))

