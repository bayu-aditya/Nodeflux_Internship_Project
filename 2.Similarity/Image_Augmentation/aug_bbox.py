import numpy as np
import cv2

from .tools_aug_bbox import rotate_im
from .tools_aug_bbox import get_enclosing_box
from .tools_aug_bbox import clip_box
from .tools_aug_bbox import get_corners
from .tools_aug_bbox import rotate_box

from .data_aug import RandomShear

def get_bbox_shear(img, bbox_img, mode):
    bbox = []
    for i in range(len(bbox_img)):
        x1 = bbox_img[i]['x1']
        y1 = bbox_img[i]['y1']
        x2 = bbox_img[i]['x2']
        y2 = bbox_img[i]['y2']
        bbox.append([x1, y1, x2, y2])
    old_bboxes = np.array(bbox)
    if (mode == 'left'):
        scale = 0.3
    elif (mode == 'right'):
        scale = -0.3
    img_, bboxes_ = RandomShear(scale)(img.copy(), old_bboxes.copy())
    return bboxes_

def get_bbox_rotate(img, bbox_img, mode):
    """
    Reference : https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
    """
    if (mode == 'left'):
        angle = 5
    elif (mode == 'right'):
        angle = -5
    bbox = []
    for i in range(len(bbox_img)):
        x1 = bbox_img[i]['x1']
        y1 = bbox_img[i]['y1']
        x2 = bbox_img[i]['x2']
        y2 = bbox_img[i]['y2']
        bbox.append([x1, y1, x2, y2])
    old_bboxes = np.array(bbox)
    
    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2
    img = rotate_im(img, angle)
    corners = get_corners(old_bboxes)
    corners = np.hstack((corners, old_bboxes[:,4:]))
    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)
    scale_factor_x = img.shape[1] / w
    scale_factor_y = img.shape[0] / h
    img = cv2.resize(img, (w,h))
    new_bbox[:,:4] = new_bbox[:,:4] / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
    bboxes  = new_bbox
    bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
    return bboxes