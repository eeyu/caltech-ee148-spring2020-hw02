import os
import numpy as np
import json
from PIL import Image
import cv2

def get_local_max(A):
    '''
    Obtains the coordinates of the local maxima in a heatmap
    '''
    # Expand the size of A by 1 on each edge
    A_aug = np.copy(A)
    (y,x) = A.shape
    A_aug = np.hstack((A_aug, np.zeros((y,1))))
    A_aug = np.hstack((np.zeros((y,1)), A_aug))
    A_aug = np.vstack((np.zeros((1,x+2)), A_aug))
    A_aug = np.vstack((A_aug, np.zeros((1,x+2))))
    
    # Compare A values with adjacent 4 values
    maxima = A > A_aug[1:-1,2:]
    maxima = np.logical_and(maxima, A > A_aug[1:-1,:-2])
    maxima = np.logical_and(maxima, A > A_aug[2:,1:-1])
    maxima = np.logical_and(maxima, A > A_aug[:-2,1:-1])
    
    b = np.where(maxima)
        
    return b

def kernel_filter(kern, img):
    '''
    Convolutes a given kernel kXxkY with a given image, size XxY
    using a FFT for speed
    @img: YxXx(Z) color image as numpy array
    @kernel: kXxkY numpy array
    
    @return: Outputs a (X)x(Y) color image as a numpy array.
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
    kern_F = np.fft.fft2(kern, s=(img_Y+kern_Y, img_X+kern_X), axes=(0,1));
    # Multiply FTs and revert.
    results_F = img_F*kern_F
    result = np.fft.ifft2(results_F, axes=(0,1));
    
    
    # Crop padding accordingly
    if(img.ndim==3):
        result = result[kern_Y//2:kern_Y//2-kern_Y,kern_X//2:kern_X//2-kern_X,:]
    else:
        result = result[kern_Y//2:kern_Y//2-kern_Y,kern_X//2:kern_X//2-kern_X]
    
    return np.real(result)

def compute_convolution(kern, img, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality.
    
    @img: YxXx(Z) image as numpy array
    @kern: kXxkYx(Z) template (unflipped)
    
    @return: Outputs an XxY heatmap
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
        
    # Pad the image. I don't know why this padding works and others don't, but
    # it worked out
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


def predict_boxes(heatmaps):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    
    Confidence taken as the inverse of variance around the maxima
    
    The heatmap may be multidimensional in Z for added functionality. The first
    indexed heatmap should be the primary one
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    global template
    global threshold
    
    box_height = template.shape[0]
    box_width = template.shape[1]

    
    thresh = 0.25
    var_radius = 5 # half-width of region to sample for variance
    
    # Extract heatmaps
    heatmap_combined = heatmaps[0]
    heatmap_orig = heatmaps[1]
    heat_height = heatmap_combined.shape[0]
    heat_width = heatmap_combined.shape[1]
    
    # Filter out values that don't meet threshold
    heatmap_combined[heatmap_combined < thresh] = thresh
    
    # Find the local maxima
    inds = get_local_max(heatmap_combined)

    for i in range(len(inds[0])):
        tl_row = int(inds[0][i]) - int(box_height/2)
        tl_col = int(inds[1][i]) - int(box_width/2)
        br_row = tl_row + box_height
        br_col = tl_col + box_width
        
        # Confidence
        center_y = tl_row+int(box_height/2)
        center_x = tl_col+int(box_width/2)
        variance = np.var(heatmap_orig[max(0,center_y-var_radius):min(heat_height,center_y+var_radius),
                                       max(0,center_x-var_radius):min(heat_width,center_x+var_radius)])
        # Normalize variance to 0,1
        conf = (variance**0.25)/35
        #print([tl_row,tl_col,br_row,br_col, conf])
        if (conf < 0):
            conf = 0
        if (conf > 1):
            conf = 1
        if (np.isnan(conf)):
            conf = 0
                        
        output.append([tl_row,tl_col,br_row,br_col, conf]) 

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 
    '''

    '''
    BEGIN YOUR CODE
    '''
    global template
    global kern_edge
    global kern_blur
    
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
        '''
        normalizes image per pixel by sum of color values
        '''
        I_brightness = np.sum(I, axis=2)
        I_brightness[I_brightness==0] = 1
        I_brightness = np.repeat(I_brightness[:,:,np.newaxis], I.shape[2], axis=2)
        I_norm = I / I_brightness
        return I_norm
    
    # First, create edge-filtered convolution
    # Blur the image and template
    I_blur = kernel_filter(kern_blur, I)
    I_edge= kernel_filter(kern_edge, I_blur)
    template_edge = kernel_filter(kern_edge, template)
    # Extract the edges
    template_edge = process_edge(template_edge,-.3)
    I_edge = process_edge(I_edge,0)
    # edge-edge convolution
    edge_convolute = compute_convolution(template_edge, I_edge)
    # Process
    edge_convolute[edge_convolute<0] = 0
    edge_convolute = edge_convolute/120 # a nice normalization constant by T&E
    
    # Convolute images as normal
    # First normalize for brightness
    I_norm = norm_sum(I)
    match_convolute = compute_convolution(template, I_norm)
    # Scale for resolution
    match_convolute = match_convolute - 155000 # a nice normalization constant by T&E
    match_convolute[match_convolute<0] = 0
    
    # Nor incorporate the edge filter
    combined = np.multiply(match_convolute, edge_convolute)
    #combined = match_convolute
    # Normalize based on kernel and image brightness
    combined = combined / np.sqrt(np.linalg.norm(template)*np.linalg.norm(I))

    # Create augmented heatmap
    heatmaps = (combined, match_convolute)
    output = predict_boxes(heatmaps)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        if not ((output[i][4] >= 0.0) and (output[i][4] <= 1.0)):
            print(output)
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

'''
Main Code
'''
# ==================================================================
# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../../data/RedLights2011_Medium'
# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 
# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

kernel_path = 'Kernels'
# get sorted list of files: 
kernel_names = sorted(os.listdir(kernel_path)) 

# Definitions of kernels
# ===================
# Template Kernel
global template
template = cv2.imread(os.path.join(kernel_path,kernel_names[0]))
template = np.asarray(template)

# Edge detection kernel
global kern_edge
kern_edge = np.ones((5,5)) * -1
kern_edge[2,2] = -np.sum(kern_edge)+1

# Blur kernel
global kern_blur
kern_blur = np.array([[1,  4,  6,  4, 1], 
                      [4, 16, 24, 16, 4],
                      [6, 24, 36, 24, 6],
                      [4, 16, 24, 16, 4],
                      [1,  4,  6,  4, 1]])
kern_blur = kern_blur/256


# =================================================================
# load splits: 
split_path = '../../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(int(len(file_names_train))):

    # read image using PIL:
    I = cv2.imread(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)
    
    #print(file_names_train[i])

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train_weak.json'),'w') as f:
    json.dump(preds_train,f)

'''
Make predictions on the test set. 
'''
preds_test = {}
for i in range(len(file_names_test)):

    # read image using PIL:
    I = cv2.imread(os.path.join(data_path,file_names_test[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_test[file_names_test[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_test_weak.json'),'w') as f:
    json.dump(preds_test,f)
