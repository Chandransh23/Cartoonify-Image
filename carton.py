# Importing the basic libraries

import cv2
import glob
import numpy as np
import pytesseract
from pprint import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14,8]

import warnings 
warnings.filterwarnings('ignore')


# Reading the image file

img = cv2.imread('picn.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plotting the image
def plot_image(img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    
plot_image(img)
plt.show()



# Filtering Noise

def filtering_noise(image):
    return cv2.bilateralFilter(image, 15, 300, 300)

noiseless_img = filtering_noise(img)

plot_image(noiseless_img, cmap='gray')
plt.show()




# Converting the image to grayscale

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_img = get_grayscale(img)

plot_image(gray_img, cmap='gray')
plt.show()


# Applying Gaussian Blur

def remove_noise(image):
    return cv2.GaussianBlur(gray_img,(5,5),1)

blur_img = remove_noise(gray_img)

plot_image(blur_img, cmap='gray')
plt.show()


# Applying Thresholding

def apply_thresholding(image):
    return cv2.adaptiveThreshold(image, 255, 
  cv2.ADAPTIVE_THRESH_MEAN_C, 
  cv2.THRESH_BINARY, 9, 9)

thresholded_img = apply_thresholding(blur_img)

plot_image(thresholded_img, cmap='gray')
plt.show()



 #Applying cartoonify filter

def cartoonify_filter(image):
    return cv2.bitwise_and(image, image, mask=thresholded_img)

cartoonify_img = cartoonify_filter(noiseless_img)

plot_image(cartoonify_img, cmap='gray')
plt.show()


## Reading multiple files from images directory

file_list = glob.glob(r'C:\Users\HP-PC\Desktop\mini project\cartoonify_image\*.jpg')
pprint(file_list)



# Plotting the images
plt.figure(figsize=[16,8])

def plot_multple_images(e, i):
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(3,5,e+1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

for e, i in enumerate(file_list):
    plot_multple_images(e, i)
    
plt.show()



def a(img_path, e):
    
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    noiseless_img = cv2.bilateralFilter(img, 15, 300, 300)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.GaussianBlur(gray_img,(5,5),1)

    thresholded_img = cv2.adaptiveThreshold(blur_img, 255, 
      cv2.ADAPTIVE_THRESH_MEAN_C, 
      cv2.THRESH_BINARY, 9, 9)

    cartoonify_img = cv2.bitwise_and(img, img, mask=thresholded_img)

    plt.subplot(1,5,e+1)
    plt.imshow(cartoonify_img)
    plt.xticks([])
    plt.yticks([])
    
    
for e,i in enumerate(file_list):
    a(i, e)

plt.show()