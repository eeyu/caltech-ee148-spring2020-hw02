import os
import numpy as np
import cv2
import time
import seaborn as sb
import pylab


'''
Processes kernels
'''

def kernel_filter_fourier(kern, img):
    '''
    Convolutes a given kernel kXxkY with a given image, size XxY
    using a FFT for speed
    @img: YxXxZ color image as numpy array
    @kernel: kXxkY numpy array
    
    @return: Outputs a (X)x(Y)xZ color image as a numpy array.
    '''
    img_Y = np.shape(img)[0]
    img_X = np.shape(img)[1]
    kern_Y = np.shape(kern)[0]
    kern_X = np.shape(kern)[1]
    kern_X2 = int(kern_X/2)
    kern_Y2 = int(kern_Y/2)
        
    # If kernel is 2d matrix, stacks to depth of image
    if (img.ndim > 2):
        img_Z = np.shape(img)[2]
        kern = np.repeat(kern[:,:,np.newaxis], img_Z, axis=2)
    
    # Do the convolution
    # Compute the FT of the image and kernel
    img_F  = np.fft.fft2(img, s=(img_Y+kern_Y, img_X+kern_X), axes=(0,1)); # add 2 because kernel FFT is weird
    #img_F = np.fft.fftshift(img_F, axes=(0,1))
    kern_F = np.fft.fft2(kern, s=(img_Y+kern_Y, img_X+kern_X), axes=(0,1));
    #kern_F = np.fft.fftshift(kern_F, axes=(0,1))
    # Multiply FTs and revert.
    results_F = img_F*kern_F
    #results_F = np.fft.ifftshift(results_F, axes=(0,1))
    result = np.fft.ifft2(results_F, axes=(0,1));
    
    
    # Crop padding accordingly
    if(img.ndim==3):
        result = result[kern_Y2:kern_Y2-kern_Y,kern_X2:kern_X2-kern_X,:]
    else:
        result = result[kern_Y2:kern_Y2-kern_Y,kern_X2:kern_X2-kern_X]
    
    return np.abs(result)

def convolute_2d_fourier(kern, img):
    '''
    Convolutes a given kernel kXxkY with a given image, size XxY
    using a FFT for speed
    @img: YxXxZ color image as numpy array
    @kernel: kXxkY numpy array or kXxkYxZ image kernel (flipped)
    
    @return: Outputs a (X)x(Y)xZ color image as a numpy array.
    '''
    # Size of image
    img_Y = np.shape(img)[0]
    img_X = np.shape(img)[1]
    kern_Y = np.shape(kern)[0]
    kern_X = np.shape(kern)[1]
    kern_X2 = kern_X//2
    kern_Y2 = kern_Y//2
        
    kern = np.flip(kern, axis=(0,1))

    # If kernel is 2d matrix, stacks to depth of image
    if (kern.ndim == 2 and img.ndim > 2):
        img_Z = np.shape(img)[2]
        kern = np.repeat(kern[:,:,np.newaxis], img_Z, axis=2)
        
    # add padding to kernel
# =============================================================================
#     kern_pad = np.zeros(img.shape)
#     pad_center = (img_Y//2, img_X//2)
#     if (kern_pad.ndim == 2):
#         kern_pad[(pad_center[0]-kern_Y2):(pad_center[0]+kern_Y-kern_Y2), 
#                  (pad_center[1]-kern_X2):(pad_center[1]+kern_X-kern_X2)] = kern
#     else:
#         kern_pad[(pad_center[0]-kern_Y2):(pad_center[0]+kern_Y-kern_Y2), 
#                  (pad_center[1]-kern_X2):(pad_center[1]+kern_X-kern_X2),:] = kern
#     kern = kern_pad
#     kern = np.fft.ifftshift(kern)
# =============================================================================
        
        # Pad Image
    edge = np.repeat(img[np.newaxis,-1,:,:], kern_Y2, axis=0)
    img = np.vstack((img, edge))
    edge = np.repeat(img[np.newaxis,0,:,:], kern_Y-kern_Y2, axis=0)
    img = np.vstack((edge, img))
    edge = np.repeat(img[:,np.newaxis,-1,:], kern_X2, axis=1)
    img = np.hstack((img, edge))
    edge = np.repeat(img[:,np.newaxis,0,:], kern_X-kern_X2, axis=1)
    img = np.hstack((edge, img))
    
    # Do the convolution
    # Compute the FT of the image and kernel
    img_F  = np.fft.fft2(img, s=(img_Y, img_X), axes=(0,1));
    kern_F = np.fft.fft2(kern, s=(img_Y, img_X), axes=(0,1));
    # Add some padding
    # img_F  = np.fft.fft2(img, s=(img_Y+kern_Y, img_X+kern_X), axes=(0,1));
    # kern_F = np.fft.fft2(kern, s=(img_Y+kern_Y, img_X+kern_X), axes=(0,1));
    # Multiply FTs and revert
    results_F = img_F*kern_F
    result = np.fft.ifft2(results_F, axes=(0,1));
    
    # Flatten into heatmap, if not already flattened
    if (img.ndim > 2):
        result = np.sum(result, axis=2)
    # Crop padding
    #result = result[kern_Y2:kern_Y2-kern_Y,kern_X2:kern_X2-kern_X]
    #result = result[kern_Y:,kern_X:]
        
    # shift result
    #result = np.fft.ifftshift(result, axes=(0,1))
        
    # I give up. Just roll it
    result = np.roll(result, (-kern_Y,-kern_X), axis=(0,1))

    
    return np.real(result)

# set the path to the downloaded data: 
data_path = '../../data/RedLights2011_Medium'
# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 
# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

kernel_path = 'Kernels'
# get sorted list of files: 
kernel_names = sorted(os.listdir(kernel_path)) 

# edge detection kernel
kern_edge = np.ones((5,5)) * -1
kern_edge[2,2] = -np.sum(kern_edge)+1

kern_blur = np.ones((3,3))
kern_blur[1,:] = 2
kern_blur[:,1] = 2
kern_blur[1,1] = 4
kern_blur = kern_blur*1/16

# larger blur
kern_blur2 = np.array([[1,  4,  6,  4, 1], 
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1,  4,  6,  4, 1]])
kern_blur2 = kern_blur2/256

kern_identity = np.zeros((9,9));
kern_identity[4,4] = 1;

# Final Run


def norm_sum(I):
    I_brightness = np.sum(I, axis=2)
    I_brightness[I_brightness==0] = 1
    I_brightness = np.repeat(I_brightness[:,:,np.newaxis], I.shape[2], axis=2)
    I_norm = I / I_brightness
    return I_norm

# Size of image
# Image Kernel
kern = cv2.imread(os.path.join(kernel_path,kernel_names[0]))
kern = np.asarray(kern)

# read an image:
i = 10
img = cv2.imread(os.path.join(data_path,file_names[i]))
img = np.asarray(img)
img = norm_sum(img)

start = time.time()
result = convolute_2d_fourier(kern, img)
print(time.time()-start)

thresh = 155000
result[result<thresh]=thresh
result = result - thresh
print(thresh)

# Convolute images as normal
# First normalize for brightness
#img_norm = norm_sum(img)
#filter_img_FFT = convolute_2d_fourier(kern_img, img_norm)

# Results
sb.heatmap(result);
#sb.heatmap(img[:,:,0])

# filter = kernel_filter_fourier(kern_identity, I)

cv2.imshow("original", img);
cv2.waitKey(0);    
cv2.destroyAllWindows()


#TODO
# account for size or trafic




