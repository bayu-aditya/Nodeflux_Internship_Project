# Author Bayu Aditya

import os
import pandas as pd
import numpy as np
import collections
import cv2
from datetime import datetime

from .datasets import image_augmentation
from .images import export_image
from .similarity import compareSIFT, compareSURF


def makehash():
    return collections.defaultdict(makehash)

def match_sift_distanceMAE(match):
    distance = []
    for i in range(len(match)):
        distance.append(match[i].distance)
    MAE = np.mean(np.abs(distance))
    return MAE

class benchmark_class():
    def __init__(self, dir_datasets, num_indeks = None):
        self.dir_datasets = dir_datasets
        self.num_index = num_indeks

    def input_only_dataset(self):
        all = {}
        list_image = os.listdir(self.dir_datasets)
        for name in list_image:
            img_label = name
            loc_img = self.dir_datasets + img_label
            img_read = cv2.imread(loc_img)
            all.update({img_label : img_read})
        self.data = all
        self.num_index_max = len(self.data)

    def input_json(self, loc_data_json):
        data_class = export_image(loc_data_json, self.dir_datasets)
        self.data = data_class.get_all_images_label()
        self.num_index_max = len(self.data)

    def summary(self):
        start_time = datetime.now()

        score_aug = makehash()
        score_diff = makehash()

        indeks = 0
        if (self.num_index == None):
            num_max = self.num_index_max
        else:
            num_max = self.num_index

        for label in self.data.keys():
            indeks += 1                              
            
            img_ori = self.data[label]
            class_img_aug = image_augmentation(img_ori)
            dict_img_aug = class_img_aug.get_data()

            for subclass in dict_img_aug.keys():
                for mode in dict_img_aug[subclass]:
                    img_aug = dict_img_aug[subclass][mode]
                    match = compareSIFT(img_ori, img_aug)
                    score = match_sift_distanceMAE(match)
                    score_aug[label][subclass][mode] = score

            for i, target in enumerate(self.data.keys()):
                img_target = self.data[target]
                match = compareSIFT(img_ori, img_target)
                score = match_sift_distanceMAE(match)
                score_diff[label][target] = score
                if (i == num_max-1):
                    break 

            end_time = datetime.now()
            print('Keys ',indeks,' has been completed. With duration : ', end_time - start_time)

            if (indeks == num_max):
                break  

        return score_aug, score_diff, num_max

def get_dataframe(score_aug, score_diff):
    param = {'get_blur' : ['very_low', 'low', 'med', 'high'],
        'get_crop' : ['very_low', 'low', 'med', 'high'],
        'get_intensity' : ['glow', 'dark'],
        'get_noise' : ['gaussian', 's&p', 'speckle'],
        'get_rotate' : ['left', 'right'],
        'get_translate' : ['up', 'down', 'right', 'left'],
        'get_scale' : ['small', 'big'],
        'get_shear' : ['left', 'right']}
    score = []
    columns_score = []
    index_score = []

    for label in score_aug.keys():
        score_label = []
        columns_score = []
        for subclass in param:
            for mode in param[subclass]:
                score_label.append(score_aug[label][subclass][mode])
                columns_score.append(subclass + ':' + mode)
                
        for col in score_diff[label]:
            score_label.append(score_diff[label][col])
            columns_score.append(col)   
        
        index_score.append(label)
        score.append(score_label)
        
    dataframeMAE = pd.DataFrame(score, columns=columns_score, index=index_score)