import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    iou = np.random.random()
    width = min(box_1[2], box_2[2]) - max(box_1[0], box_2[0]);
    height = min(box_1[3], box_2[3]) - max(box_1[1], box_2[1]);
    
    # Boxes don't intersect
    if width<0 or height<0:
        iou = 0
        return iou
    
    # Boxes intersect. Continue
    intersect = width * height;
    
    area_b1 = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
    area_b2 = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])
    union = area_b1 + area_b2 - intersect
    
    iou = intersect/union
    
    if (iou < 0):
        iou = 0
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    associated = [] # list of predictions that have already been associated
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            iou_max = iou_thr
            best_pred = -1
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                conf = pred[j][4]
                # Check if object can be associated, and is not already associated
                # if iou greater than max, greater than thresh
                if (iou > iou_max and conf > conf_thr and j not in associated): 
                    iou_max = iou
                    best_pred = j
            if best_pred != -1: # An object was correctly detected - true positive
                TP = TP+1
                associated.append(j)
            else: # No detection made - false negative
                FN = FN+1

                
    # Count total number of predictions meeting threshold
    P = 0
    for pred_file, pred in preds.items():
        for j in range(len(pred)):
            conf = pred[j][4]
            if conf > conf_thr:
                P = P+1
                
    # False positive: total positive - true positives
    FP = P - TP

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

global plot_PR_graph
def plot_PR_graph(use_train=True, use_weak=False, thresh=0.5):
    ''' Load in data to gts and preds '''
    if use_weak:
        with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
            gts_train = json.load(f)
        with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
            gts_test = json.load(f)
        gts = {**gts_train, **gts_test}
        with open(os.path.join(preds_path,'preds_train_weak.json'),'r') as f:
            preds_train = json.load(f)
        with open(os.path.join(preds_path,'preds_test_weak.json'),'r') as f:
            preds_test = json.load(f)
        preds = {**preds_train, **preds_test}
    else:
        if use_train:
            with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
                gts = json.load(f)
            with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
                preds = json.load(f)
        else:
            with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
                gts = json.load(f)
            with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
                preds = json.load(f)
 
    # Load in confidence values
    confidence_thrs = []
    for fname in preds:
        for i in range(len(preds[fname])):
            pred = preds[fname][i]
            confidence_thrs.append(np.array([pred[4]], dtype=float))
    # Compute the counds
    tp = np.zeros(len(confidence_thrs))
    fp = np.zeros(len(confidence_thrs))
    fn = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp[i], fp[i], fn[i] = compute_counts(preds, gts, iou_thr=thresh, conf_thr=conf_thr)
    
    # Plot training set PR curves
    precision = (tp / (fp + tp))# true/total predictions
    recall = (tp / (fn + tp)) # detected/total objects
    inds = np.lexsort((precision, recall))
    plot = [recall[inds],precision[inds]]
    plt.plot(plot[0][:], plot[1][:], label=thresh)

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../../data/hw02_annotations'

# load splits:
split_path = '../../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))


plot_PR_graph(use_train=True, use_weak=True, thresh=0.75)
plot_PR_graph(use_train=True, use_weak=True, thresh=0.5)
plot_PR_graph(use_train=True, use_weak=True, thresh=0.25)
plot_PR_graph(use_train=True, use_weak=True, thresh=0.01)
plt.legend(loc="bottom left")
plt.xlabel('R')
plt.ylabel('P')
plt.show()




# =============================================================================
# '''
# Load data. 
# '''
# with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
#     preds_train = json.load(f)
#     
# with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
#     gts_train = json.load(f)
# 
# with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
#     preds_test = json.load(f)
#     
# with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
#     gts_test = json.load(f)
# 
# # For a fixed IoU threshold, vary the confidence thresholds.
# # The code below gives an example on the training set for one IoU threshold. 
# confidence_thrs = []
# for fname in preds_train:
#     for i in range(len(preds_train[fname])):
#         pred = preds_train[fname][i]
#         confidence_thrs.append(np.array([pred[4]], dtype=float))
# tp_train = np.zeros(len(confidence_thrs))
# fp_train = np.zeros(len(confidence_thrs))
# fn_train = np.zeros(len(confidence_thrs))
# for i, conf_thr in enumerate(confidence_thrs):
#     tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.01, conf_thr=conf_thr)
# 
# # Plot training set PR curves
# precision = (tp_train/(fp_train+tp_train))# true/total predictions
# recall = (tp_train/(fn_train+tp_train)) # detected/total objects
# inds = np.lexsort((precision, recall))
# plot = [recall[inds],precision[inds]]
# plt.plot(plot[0][:], plot[1][:])
# 
# =============================================================================



             
