import os
import numpy as np
import json
import cv2


global plot_boxes
def plot_boxes(inds, group):
    ''' 
    Plots image i with bounding boxes from predictions
    '''
    for i in inds:
        name = file_names_train[i]
        I = cv2.imread(os.path.join(data_path,name))
        
        # Obtain the associated predictions
        pred_current = preds_train[name]
        
        
        # Show all bounding boxes
        for j in range(len(pred_current)):
            rect = pred_current[j]
            conf = rect[4]
            # Add the rectangle to the image
            I = cv2.rectangle(I,(rect[1],rect[0]),(rect[3],rect[2]),(int((1-conf)*255),0,int(conf*255)),1)
        
        # save the image
        #cv2.imwrite(pictures_path + '/'+group+'_'+file_names[i], I );
        
        # Display the image
        cv2.imshow(group + ": " + name, I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


data_path = '../../data/RedLights2011_Medium'
preds_path = '../data/hw02_preds'

split_path = '../../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))
# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# path for picture saving
pictures_path = '../results'
os.makedirs(pictures_path,exist_ok=True) # create directory if needed 


with open(os.path.join(preds_path,'preds_train_weak.json'),'r') as f:
        preds_train = json.load(f)
        
        
plot_boxes(range(0,30), "good")
# plot_boxes([10, 46, 35, 51, 109], "good")
# plot_boxes([11, 40, 38, 71, 77], "misdetection")
# plot_boxes([31, 68, 85, 51], "bad")

