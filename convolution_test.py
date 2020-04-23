import os
import numpy as np
import cv2
import time
import seaborn as sb

'''
Processes kernels
'''

def convolute_2d_regular(kernel, img):
    '''
    Convolutes a given kernel kXxkY with a given image, size XxY
    Out of bound regions are given the average value of the in-bounds dot product
    Uses edge extension for edge handling
    @img: XxYx3 color image as numpy array
    @kernel: kXxkY numpy array. Lengthes must be odd
    
    @return: Outputs a (X)x(Y)x3 color image as a numpy array.
    
    '''
    # Edge and half-edge length of kernel
    kX = np.shape(kernel)[0]
    kY = np.shape(kernel)[1]
    kX2 = (np.shape(kernel)[0]-1)/2
    kY2 = (np.shape(kernel)[1]-1)/2
    # Size of image
    X = np.shape(img)[0]
    Y = np.shape(img)[1]
    
    # Extends image by repeating each edge by kX2 and kY2 pixels
    img_ext = np.copy(img)
    
    edge = np.repeat(img_ext[np.newaxis,X-1,:,:], kY2, axis=0)
    img_ext = np.vstack((img_ext, edge))
    
    edge = np.repeat(img_ext[np.newaxis,0,:,:], kY2, axis=0)
    img_ext = np.vstack((edge, img_ext))
    
    edge = np.repeat(img_ext[:,np.newaxis,Y-1,:], kX2, axis=1)
    img_ext = np.hstack((img_ext, edge))
    
    edge = np.repeat(img_ext[:,np.newaxis,0,:], kX2, axis=1)
    img_ext = np.hstack((edge, img_ext))
    
    # Do the convolution
    # Convert kernel to 1d array
    kernel_1d = kernel.reshape(kX*kY)
    
    # initialize output
    filtered_img = np.zeros((X,Y,3))
    
    for x in range(X):        
        for y in range(Y):
            # The relevant portion of the image
            img_extract = img_ext[x:x+kX, y:y+kY, :]
            img_extract_1d = img_extract.reshape((kX*kY), 3)
            for c in range(3): # rbg color
                filtered_img[x,y,c] = np.dot(kernel_1d, img_extract_1d[:,c])
    
    return filtered_img

def matched_filter_convolution_2d(kernel, img):
    '''
    Convolutes a given kernel kXxkY with a given image, size XxY
    Convolution only applied in regions where kernel fully fits in image
    @img: color image as numpy array, kXxkYx3
    @kernel: color image as numpy array, XxYx3
    
    @return: Outputs a (X-kX)x(Y-kY) numpy array. Each point corresponds to the result
    where the top left corner of the kernel was placed in the original image.
    
    '''
    # Find size of image and kernel
    kX = np.shape(kernel)[0]
    kY = np.shape(kernel)[1]
    X = np.shape(img)[0]
    Y = np.shape(img)[1]
    
    # Convert kernel to 1d array
    kernel_1d = kernel.reshape((kX*kY, 3))
    
    # initialize output
    filtered_img = np.zeros((X-kX+1,Y-kY+1))
    
    # do the convolution
    for x in range(X-kX+1):
        for y in range(Y-kY+1):
            # The relevant portion of the image
            img_extract = img[x:x+kX, y:y+kY, :]
            img_extract_1d = img_extract.reshape((kX*kY), 3)
            # normalize
            #img_extract_1d = img_extract_1d/np.linalg.norm(img_extract_1d) # norm
            img_extract_1d = img_extract_1d/np.sum(img_extract_1d) # sum
            for c in range(3): # rbg color
                filtered_img[x,y] += np.dot(kernel_1d[:,c], img_extract_1d[:,c])
    
    return filtered_img

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
        result = result[int(kern_Y/2):int(kern_Y/2)-kern_Y,int(kern_X/2):int(kern_X/2)-kern_X,:]
    else:
        result = result[int(kern_Y/2):int(kern_Y/2)-kern_Y,int(kern_X/2):int(kern_X/2)-kern_X]
    
    return np.real(result)

def convolute_2d_fourier(kern, img, flatten=True, correlate=False):
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
        
    if (img.ndim == 3):
        # Pad Image
        edge = np.repeat(img[np.newaxis,-1,:,:], kern_Y2, axis=0)
        img = np.vstack((img, edge))
        edge = np.repeat(img[np.newaxis,0,:,:], kern_Y-kern_Y2, axis=0)
        img = np.vstack((edge, img))
        edge = np.repeat(img[:,np.newaxis,-1,:], kern_X2, axis=1)
        img = np.hstack((img, edge))
        edge = np.repeat(img[:,np.newaxis,0,:], kern_X-kern_X2, axis=1)
        img = np.hstack((edge, img))
    else:
        edge = np.repeat(img[np.newaxis,-1,:], kern_Y2, axis=0)
        img = np.vstack((img, edge))
        edge = np.repeat(img[np.newaxis,0,:], kern_Y-kern_Y2, axis=0)
        img = np.vstack((edge, img))
        edge = np.repeat(img[:,np.newaxis,-1], kern_X2, axis=1)
        img = np.hstack((img, edge))
        edge = np.repeat(img[:,np.newaxis,0], kern_X-kern_X2, axis=1)
        img = np.hstack((edge, img))
    
    # Do the convolution
    # Compute the FT of the image and kernel
    img_F  = np.fft.fft2(img, s=(img_Y, img_X), axes=(0,1));
    kern_F = np.fft.fft2(kern, s=(img_Y, img_X), axes=(0,1));
    # Multiply FTs and revert
    results_F = img_F*kern_F
    result = np.fft.ifft2(results_F, axes=(0,1));
    
    # Flatten into heatmap, if not already flattened
    if (img.ndim > 2):
        result = np.sum(result, axis=2)

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



# Image Kernel
kern_img = cv2.imread(os.path.join(kernel_path,kernel_names[0]))
# convert to numpy array:
kern_img = np.asarray(kern_img)

# edge detection kernel
kern_edge = np.ones((5,5)) * -1
kern_edge[2,2] = -np.sum(kern_edge)+1

# =============================================================================
# kern_edge = np.zeros((3,3)) 
# kern_edge[:,2] = -1
# kern_edge[2,:] = -1
# kern_edge[2,2] = -np.sum(kern_edge)+1
# 
# =============================================================================
# summing kernel
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



# Regular Kernel Speed Test (30 times faster)
# =============================================================================
# # apply kernel, using regular method
# start = time.time()
# filter_reg = convolute_2d_regular(kern_edge, I)
# print(time.time()-start)
# start = time.time()
# #apply kernel, using fourier
# filter_FFT = convolute_2d_fourier(kern_edge, I,flatten=False)
# print(time.time()-start)
# 
# cv2.imshow("reg", filter_reg);
# cv2.imshow("FFT", filter_FFT);
# cv2.waitKey(0);    
# cv2.destroyAllWindows()
# =============================================================================

# Image Kernel Speed Test (100 times faster)
# =============================================================================
# # apply kernel, using regular method
# start = time.time()
# filter_img_FFT = convolute_2d_fourier(kern_img, I_norm, correlate=True)
# print(time.time()-start)
# start = time.time()
# # apply kernel, using regular method
# filter_img_reg = matched_filter_convolution_2d(kern_img, I)
# print(time.time()-start)
# 
# sb.heatmap(filter_img_FFT);
# sb.heatmap(filter_img_reg);
# =============================================================================

# Testing Filtering Methods

# Normalization Tests
# =============================================================================
# # first normalize brightness by averaging all pixel intensities
# I_brightness = np.sum(I, axis=2)
# I_brightness = np.repeat(I_brightness[:,:,np.newaxis], I.shape[2], axis=2)
# I_norm = I / I_brightness
# 
# # view normalization
# cv2.imshow("Original", I);
# cv2.imshow("Normalization", I_norm);
# cv2.waitKey(0);    
# cv2.destroyAllWindows()
# =============================================================================

# Edge Detection Tests
# =============================================================================
# # VH detection kernel
# kern_edgeVH = np.zeros((5,5))
# kern_edgeVH[1:4,:] = -1
# 
# kern_edgeVH[:,1:4] = -1
# kern_edgeVH[2,2] = -np.sum(kern_edgeVH)+1
# 
# # Square Boxes kernel
# w = 19
# c = w
# kern_lines = np.zeros((2*w,2*w))
# for i in range(w):
#     kern_lines[c-w+i:c+w-i,c-w+i:c+w-i]=i%2
# =============================================================================

# view normalization
# cv2.imshow("edge", ker_kernel_edge);
# cv2.waitKey(0);    
# cv2.destroyAllWindows()


def process_edge(image_edge, neg_val=0):
    '''
    extracts the edges of edge-filtered image
    '''
    flat_edge_orig = np.sum(image_edge, axis=2)
    flat_edge = np.copy(flat_edge_orig)
    flat_edge[flat_edge_orig<0] = 1
    flat_edge[flat_edge_orig>0] = neg_val
    return flat_edge

def norm_sum(I):
    I_brightness = np.sum(I, axis=2)
    I_brightness[I_brightness==0] = 1
    I_brightness = np.repeat(I_brightness[:,:,np.newaxis], I.shape[2], axis=2)
    I_norm = I / I_brightness
    return I_norm

def normalize_range(I):
    I = I - np.min(I)
    I = I/np.max(I)
    return(I)

# =============================================================================
# # Testing

# # read an image:
# i = 11
# I = cv2.imread(os.path.join(data_path,file_names[i]))
# # convert to numpy array:
# I = np.asarray(I)
# I_edge= kernel_filter_fourier(kern_edge, I)
# ker_kernel_edge = kernel_filter_fourier(kern_edge, kern_img)
# # =============================================================================
# # I_edge= convolute_2d_regular(kern_edge, I)
# # ker_kernel_edge = convolute_2d_regular(kern_edge, kern_img)
# # =============================================================================
# ker_kernel_edge = process_edge(ker_kernel_edge,0)
# I_edge = process_edge(I_edge,0)
# 
# # Now blur
# # =============================================================================
# # ker_kernel_edge = kernel_filter_fourier(kern_blur, ker_kernel_edge)
# # I_edge = kernel_filter_fourier(kern_blur, I_edge)
# # =============================================================================
# 
# #sb.heatmap(ker_kernel_edge)
# #sb.heatmap(I_edge)
# 
# 
# #flat_edge_img = process_edge(I_edge)
# #ker_kernel_edge = process_edge(ker_kernel_edge)
# #sb.heatmap(flat_edge_img);
# 
# # edge-edge convolution, image-based
# edge_filter = convolute_2d_fourier(ker_kernel_edge, I_edge, correlate=True)
# # =============================================================================
# # edge_filter_alt = edge_filter
# # edge_filter_alt[edge_filter_alt<1] = 1
# # thresh_l = np.max(edge_filter_alt)/10
# # thresh_u = np.max(edge_filter_alt)
# # edge_filter_alt[edge_filter_alt<thresh_l] = thresh_l
# # edge_filter_alt[edge_filter_alt>thresh_u] = thresh_u
# # edge_filter_alt -= thresh_l-1
# # =============================================================================
# # =============================================================================
# # edge_filter_alt = np.copy(edge_filter)
# # edge_filter_alt[edge_filter_alt<1] = 1
# # thresh = np.max(edge_filter_alt)/6
# # edge_filter_alt[edge_filter_alt>thresh] = thresh
# # =============================================================================
# edge_filter = kernel_filter_fourier(kern_blur, edge_filter)
# edge_filter = edge_filter/np.max(edge_filter)
# 
# 
# #sb.heatmap(ker_kernel_edge)
# 
# # =============================================================================
# # I_ident = convolute_2d_fourier(kern_identity, I, flatten=False)
# # cv2.imshow("identity", I_ident);
# # cv2.imshow("original", I);
# # # cv2.imshow("edge", I_edge);
# # cv2.waitKey(0);    
# # cv2.destroyAllWindows()
# # =============================================================================
# 
# =============================================================================


# Final Run
# read an image:
i = 15
I = cv2.imread(os.path.join(data_path,file_names[i]))
I = np.asarray(I)

# Edge filter the images and extract edges
I_blur = kernel_filter_fourier(kern_blur2, I)
I_edge= kernel_filter_fourier(kern_edge, I_blur)
ker_kernel_edge = kernel_filter_fourier(kern_edge, kern_img)
I_edge = process_edge(I_edge,0)
ker_kernel_edge = process_edge(ker_kernel_edge,-.3)

kernel_norm = np.linalg.norm(kern_img)
# edge-edge convolution
edge_filter = convolute_2d_fourier(ker_kernel_edge, I_edge)
edge_filter[edge_filter<0] = 0
edge_filter = edge_filter/120

# Convolute images as normal
# First normalize for brightness
I_norm = norm_sum(I)
filter_img_FFT = convolute_2d_fourier(kern_img, I_norm)
filter_img_FFT = filter_img_FFT - 155000
filter_img_FFT[filter_img_FFT<0] = 0

# Nor incorporate the edge filter
combined = np.multiply(filter_img_FFT, edge_filter)
combined = combined / np.sqrt(kernel_norm*np.linalg.norm(I))

# Results
# view threshold
threshed = np.copy(combined)
thresh = 0.2
threshed[threshed<thresh] = 0
#sb.heatmap(threshed)
#sb.heatmap(normalize_range(combined)); # nice view
#sb.heatmap(combined);
#sb.heatmap(filter_img_FFT);
#sb.heatmap(edge_filter);

#cv2.imshow("ker_edge", ker_kernel_edge);
# fft = np.fft.fft2(I,axes=(0,1))
# ifft = np.zeros(shape=fft.shape, dtype=np.uint8)
# ifft = np.real(np.fft.ifft2(fft,axes=(0,1))).astype(np.uint8)
identity = kernel_filter_fourier(kern_identity, I)

# cv2.imshow("original", I);
# cv2.imshow("id", identity.astype(np.uint8));

# #cv2.imshow("edge", I_edge);
# cv2.waitKey(0);    
# cv2.destroyAllWindows()

#TODO
# account for size or trafic




