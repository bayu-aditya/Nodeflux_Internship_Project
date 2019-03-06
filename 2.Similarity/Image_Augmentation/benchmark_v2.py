# Author : Bayu Aditya

import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from .datasets import image_augmentation
from .similarity import compareSURF, compareSIFT, compareORB

class benchmark_class():
    def __init__(self, dir_dataset, mode = 'SIFT'):
        self.dir_dataset = dir_dataset
        self.mode = mode
        self.param = {'get_blur' : ['very_low', 'low', 'med', 'high'],
        'get_crop' : ['very_low', 'low', 'med', 'high'],
        'get_intensity' : ['glow', 'dark'],
        'get_noise' : ['gaussian', 's&p', 'speckle'],
        'get_rotate' : ['left', 'right'],
        'get_translate' : ['up', 'down', 'right', 'left'],
        'get_scale' : ['small', 'big'],
        'get_shear' : ['left', 'right']}
        print("folder 'result_benchmark' must be exist in this directory !")


    def _compare_image(self, img1, img2, mode = 'SIFT'):
        if (mode == 'SIFT'): 
            matches = compareSIFT(img1, img2)
        elif (mode == 'SURF'):
            matches = compareSURF(img1, img2)
        elif (mode == 'ORB'):
            matches = compareORB(img1, img2)
        MAE = self._distance_matches_MAE(matches)
        return MAE

    @staticmethod
    def _distance_matches_MAE(matches):
        # Scoring MAE
        distance = []
        for i in range(len(matches)):
            distance.append(matches[i].distance)
        MAE = np.mean(np.abs(distance))
        return MAE

    def _compare_1_images_for_all_images(self, img_key, name_key, image_list, name_list):#, maks):
        score_raw = []
        column_list = []
        j = 0                                       # Can be removed
        for img_, name_ in zip(image_list, name_list):
            j += 1                                  # Can be removed
            #print(j)
            score = self._compare_image(img_key, img_, self.mode)
            score_raw.append(score)
            column_list.append(name_)
            #if (j == maks):                          # Can be removed
            #    break                               # Can be removed
        
        # ACCURACY metrics
        score_raw = np.array(score_raw)
        loc_min = np.where(score_raw == score_raw.min())[0][0]
        name_min = name_list[loc_min]
        #print('lokasi nilai terendah : ', loc_min)
        #print('name nilai terendah : ', name_min)
        accuracy = (1 if (name_key == name_min) else 0)
        
        column_list.append('accuracy')
        score_raw = np.append(score_raw, accuracy)

        return column_list, score_raw

    def get_result_dataframe(self, start = 1, maks = None):
        image_list, name_list = self._generate_image(self.dir_dataset)
        
        if (maks == None):
            num_maks = len(image_list)
        else:
            num_maks = maks

        raw = []
        score_all = []

        j = start-1                                  
        # For Original Image from (raw start-th) until (raw num_max-th)
        for img_, name_ in zip(image_list[start-1:num_maks], name_list[start-1:num_maks]):
            j += 1                                  
            start_time = datetime.now()             # Checkpoint time duration

            image_list_all = image_list
            name_list_all = name_list

            raw.append(name_)
            column, score_raw = self._compare_1_images_for_all_images(img_, name_, image_list_all, name_list_all)
            score_all.append(score_raw)
            print(j, self.mode, ' images original success')

        # For Augmented Image
            dict_aug = self._image_augmented(img_)
            for subclass in self.param:
                for mode in self.param[subclass]:
                    name_aug = name_[:-4] + ' ' + subclass + ' ' + mode
                    raw.append(name_aug)
                    img_ = dict_aug[subclass][mode]
                    column, score_raw = self._compare_1_images_for_all_images(img_, name_, image_list_all, name_list_all)
                    score_all.append(score_raw)
                    print('   ' + subclass + ' ' + mode + ' success')

        # create checkpoint dataframe
            # Remove checkpint from previous iteration
            if os.path.exists('result_benchmark/' + self.mode + '_' + str(start) + '_' + str(j-1) + '.csv'):
                os.remove('result_benchmark/' + self.mode + '_' + str(start) + '_' + str(j-1) + '.csv')
                print('     File ' + self.mode + '_' + str(j-1) + ' has been removed')

            column_list = column
            raw_list = raw
            df_chk = pd.DataFrame(data = score_all, index = raw_list, columns = column_list)
            df_chk.to_csv('result_benchmark/' + self.mode + '_' + str(start) + '_' + str(j) + '.csv')

        # Check duration each iteration
            end_time = datetime.now()
            print('     Duration : {} in this iteration'.format(end_time - start_time))
            print('     Operation from image {} until image {}'.format(start, maks) + '\n')

            if (j == maks):
                break

        column_list = column
        raw_list = raw
        df = pd.DataFrame(data = score_all, index = raw_list, columns = column_list)

        return df
                    
    @staticmethod
    def _generate_image(dir_dataset):
        image_list = []
        name_list = []
        list_name = os.listdir(dir_dataset)

        for name in list_name:
            if (name[-4:] == '.jpg'):
                img_ = cv2.imread(dir_dataset + name)
                image_list.append(img_)
                name_list.append(name)
        
        # Format : image_list[integer] is image_array_i-th
        return image_list, name_list

    @staticmethod
    def _image_augmented(image):
        class_image_augmented = image_augmentation(image)
        dict_image_augmented = class_image_augmented.get_result()
        return dict_image_augmented



class proses_dataframe_benchmark:
    def __init__(self, dataframe):
        self.dataframe = dataframe


    def get_z_transformation_score(self):
        """
        Menormalisasi nilai Benchmark berdasarkan transformasi Z di dalam statistika
        """
        df = self.dataframe
        data_z = df.drop(["accuracy", 'Unnamed: 0', 'Unnamed: 0.1'], axis = 1)
        
        data_z = data_z.apply((lambda x : (x - x.mean())/x.std() ))
        
        dataframe = pd.concat([df['Unnamed: 0'], df['Unnamed: 0.1'], data_z, df["accuracy"]], axis = 1)
        return dataframe


    def get_accuracy_top_3(self):
        """
        Menghasilkan kolom untuk skor akurasi 3 tertinggi, terletak di kolom terakhir
        """
        df = self.dataframe
        accuracy_top_3 = []

        for i in range(len(df)):
            score = df.drop(["accuracy", 'Unnamed: 0', 'Unnamed: 0.1'], axis = 1).iloc[i]
            score_sort = score.sort_values(ascending = True)

            name_top_3 = score_sort.index[0:3]
            label = self.name_key(df['Unnamed: 0.1'][i]) + '.jpg'

            accuracy_top_3.append(
                1.0 if (label in name_top_3) else 0.0
            )

        accuracy_top_3_df = pd.DataFrame(data = accuracy_top_3, columns = ['accuracy_top_3'])
        dataframe = pd.concat([df, accuracy_top_3_df], axis = 1)
        return dataframe


    @staticmethod
    def name_key(name):
        """
        Memisahkan nama key dari nama augmentasi
        """
        i = 0
        for j in name:
            if (j == ' ') or (j == '.'):
                break
            i += 1
        return name[:i]