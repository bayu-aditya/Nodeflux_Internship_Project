# Author : Bayu Aditya

import cv2
import matplotlib.pyplot as plt

from image_modules_v2.bbox_util import *

from image_modules_v2.CORE_PaperSpace import *
from image_modules_v2.CORE_ImageModule_v2 import produce_image

def convert_dict(dict_bbox_index):
    """
    convert :
    ---------
    data_index['bbox_char'][no_alphabet]['x1'] -----> list_loc_bbox[i][j]
                                                 |--> list_label[i]
    """
    list_loc_bbox = []
    list_label = []

    num_label = len(dict_bbox_index['bbox_char'])
    for i in range(num_label):
        label_i = dict_bbox_index['bbox_char'][i]['label']
        x1 = dict_bbox_index['bbox_char'][i]['x1']
        y1 = dict_bbox_index['bbox_char'][i]['y1']
        x2 = dict_bbox_index['bbox_char'][i]['x2'] 
        y2 = dict_bbox_index['bbox_char'][i]['y2']

        list_label.append(label_i)
        list_loc_bbox.append([x1, y1, x2, y2, i])

    list_loc_bbox = np.array(list_loc_bbox)

    return list_loc_bbox, list_label

def invert_dict(list_loc_bbox, list_label):
    """
    convert :
    ---------
    list_loc_bbox[i][j] -------> data_indeks['bbox_char'][no_alphabet]['x1']
    list_label[i] ---------|
    """
    dict_bbox_index = {
        'bbox_char' : []
        }

    num_label = len(list_loc_bbox)
    for i in range(num_label):
        x1 = list_loc_bbox[i][0]
        y1 = list_loc_bbox[i][1]
        x2 = list_loc_bbox[i][2]
        y2 = list_loc_bbox[i][3]
        j = list_loc_bbox[i][4]
        dict_bbox_index['bbox_char'].append({
            'label' : list_label[j],
            'x1' :  x1, 'y1' : y1, 'x2' : x2, 'y2' : y2})
    return dict_bbox_index

def display_augmentation_result(img_, dict_bbox_index_, title = None):
    """To show the result of image augmentation

    Parameters
    ----------
    img_ : array image augmentation
    dict_bbox_index : dictionary for bounding box with JSON style

    """
    list_loc_bbox, list_label = convert_dict(dict_bbox_index_)

    num_label = len(list_loc_bbox)
    img_target = img_.copy()
    for i in range(num_label):
        x1 = list_loc_bbox[i][0]
        y1 = list_loc_bbox[i][1]
        x2 = list_loc_bbox[i][2]
        y2 = list_loc_bbox[i][3]
        cv2.rectangle(img_target, (x1,y1), (x2, y2), [0,255,0], 1)
    
    plt.imshow(img_target)
    plt.title(title)
    plt.show()

def image_augmentation_scale_with_bbox(img_array, dict_bbox_index, mode = 'small'):
    """
    Parameters
    ----------
    img_array : array image from cv2.imread
    dict_bbox_index : dictionary for bounding box with JSON style
    mode : 'small' or 'big'

    Returns
    -------
    img_ : array image augmentation
    dict_bbox_index_ : dictionary for bounding box with JSON style from image augmentation
    """
    # Convert bbox to array
    list_loc_bbox, list_label = convert_dict(dict_bbox_index)

    # Image Augmentation Process (need mode and bbox array)
    if (mode == 'small'):
        scale = -0.2
    elif (mode == 'big'):
        scale = 0.2
    img_, bboxes_ = RandomScale(scale, different_ratio = False)(img_array.copy(), list_loc_bbox.copy())

    # Invert bbox from array to dictionary
    dict_bbox_index_ = invert_dict(bboxes_, list_label)
    return img_, dict_bbox_index_

def image_augmentation_shear_with_bbox(img_array, dict_bbox_index, mode = 'left'):
    """
    Parameters
    ----------
    img_array : array image from cv2.imread
    dict_bbox_index : dictionary for bounding box with JSON style
    mode : 'left' or 'right'


    Returns
    -------
    img_ : array image augmentation
    dict_bbox_index_ : dictionary for bounding box with JSON style from image augmentation
    """
    # Convert bbox to array
    list_loc_bbox, list_label = convert_dict(dict_bbox_index)

    # Image Augmentation Process (need mode and bbox array)
    if (mode == 'left'):
        scale_shear = 0.2
    elif (mode == 'right'):
        scale_shear = -0.2
    img_, bboxes_ = RandomShear(scale_shear)(img_array.copy(), list_loc_bbox.copy())

    # Invert bbox from array to dictionary
    dict_bbox_index_ = invert_dict(bboxes_, list_label)
    return img_, dict_bbox_index_

def image_augmentation_rotate_with_bbox(img_array, dict_bbox_index, mode = 'left'):
    """
    Parameters
    ----------
    img_array : array image from cv2.imread
    dict_bbox_index : dictionary for bounding box with JSON style
    mode : angle rotation for image augmentation

    Returns
    -------
    img_ : array image augmentation
    dict_bbox_index_ : dictionary for bounding box with JSON style from image augmentation
    """
    # Convert bbox to array
    list_loc_bbox, list_label = convert_dict(dict_bbox_index)

    # Image Augmentation Process (need mode and bbox array)
    if (mode == 'left'):
        angle = 10
    elif (mode == 'right'):
        angle = -10
    img_, bboxes_ = RandomRotate(angle)(img_array.copy(), list_loc_bbox.copy())

    # Invert bbox from array to dictionary
    dict_bbox_index_ = invert_dict(bboxes_, list_label)
    return img_, dict_bbox_index_

def image_augmentation_translate_with_bbox(img_array, dict_bbox_index, mode = 'right'):
    """
    Parameters
    ----------
    img_array : array image from cv2.imread
    dict_bbox_index : dictionary for bounding box with JSON style
    mode : 'up', 'down', 'left', 'right'

    Returns
    -------
    img_ : array image augmentation
    dict_bbox_index_ : dictionary for bounding box with JSON style from image augmentation
    """
    # Convert bbox to array
    list_loc_bbox, list_label = convert_dict(dict_bbox_index)

    # Image Augmentation Process (need mode and bbox array)
    if (mode == 'up'):
        scale_x, scale_y = 0, -0.2
    elif (mode == 'down'):
        scale_x, scale_y = 0, 0.2
    elif (mode == 'left'):
        scale_x, scale_y = -0.1, 0
    elif (mode == 'right'):
        scale_x, scale_y = 0.1, 0
    img_, bboxes_ = RandomTranslate(scale_x, scale_y)(img_array.copy(), list_loc_bbox.copy())

    # Invert bbox from array to dictionary
    dict_bbox_index_ = invert_dict(bboxes_, list_label)
    return img_, dict_bbox_index_

def image_augmentation_intensity_with_bbox(img_array, dict_bbox_index, mode = 'glow'):
    """
    Parameters
    ----------
    img_array : array image from cv2.imread
    dict_bbox_index : dictionary for bounding box with JSON style
    mode : 'glow' or 'dark

    Returns
    -------
    img_ : array image augmentation
    dict_bbox_index_ : dictionary for bounding box with JSON style from image augmentation
    """
    # Bounding not change in intensity augmentation
    list_loc_bbox, list_label = convert_dict(dict_bbox_index)
    dict_bbox_index_ = invert_dict(list_loc_bbox, list_label)

    # Image Augmentation
    produce_image_class = produce_image(img_array)
    img_ = produce_image_class.get_intensity(mode)

    return img_, dict_bbox_index_

def image_augmentation_noise_with_bbox(img_array, dict_bbox_index, mode = 'gaussian'):
    """
    Parameters
    ----------
    img_array : array image from cv2.imread
    dict_bbox_index : dictionary for bounding box with JSON style
    mode : 'gaussian', 's&p', 'speckle'

    Returns
    -------
    img_ : array image augmentation
    dict_bbox_index_ : dictionary for bounding box with JSON style from image augmentation
    """
    # Bounding not change in intensity augmentation
    list_loc_bbox, list_label = convert_dict(dict_bbox_index)
    dict_bbox_index_ = invert_dict(list_loc_bbox, list_label)

    # Image Augmentation
    produce_image_class = produce_image(img_array)
    img_ = produce_image_class.get_noise(mode)

    return img_, dict_bbox_index_

def image_augmentation_blur_with_bbox(img_array, dict_bbox_index, mode = 'med'):
    """
    Parameters
    ----------
    img_array : array image from cv2.imread
    dict_bbox_index : dictionary for bounding box with JSON style
    mode : 'very_low', 'low', 'med', 'high'

    Returns
    -------
    img_ : array image augmentation
    dict_bbox_index_ : dictionary for bounding box with JSON style from image augmentation
    """
    # Bounding not change in intensity augmentation
    list_loc_bbox, list_label = convert_dict(dict_bbox_index)
    dict_bbox_index_ = invert_dict(list_loc_bbox, list_label)

    # Image Augmentation
    produce_image_class = produce_image(img_array)
    img_ = produce_image_class.get_blur(mode)

    return img_, dict_bbox_index_