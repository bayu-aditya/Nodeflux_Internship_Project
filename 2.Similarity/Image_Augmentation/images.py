# Author : Bayu Aditya

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import cv2

import skimage.util as skutil
import skimage.transform as sktrans

class export_image():
    """
    Class untuk menampilkan data gambar dari suatu folder Dataset dan file JSON

    Input : Lokasi data JSON
    Output : Lokasi dataset
    """
    def __init__(self, loc_data_json, dir_dataset):
        self.dir_dataset = dir_dataset
        with open(loc_data_json) as f:
            self.data = json.load(f)
        self._index = 0
    
    # Menghasilkan matrix gambar
    def get_images(self):
        self._index += 1
        self.name = self.data[str(self._index)]['name']
        loc_img = self.dir_dataset + self.name
        img_bgr = cv2.imread(loc_img)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.img_rgb

    # Menghasilkan dictionary yang berisi semua gambar beserta labelnya
    def get_all_images_label(self):
        all = {}
        for indeks in self.data.keys():
            img_label = self.data[indeks]['name']
            loc_img = self.dir_dataset + img_label
            img_read = cv2.imread(loc_img)
            all.update({img_label : img_read})
        return all
    
    # Menampilkan nama gambar
    def show_label(self):
        return self.name
    
    # Menampilkan Gambar
    def show_images(self):
        plt.imshow(self.img_rgb)
        plt.show()

class produce_image():
    """
    Class untuk memproses data gambar RGB dari suatu input
    """
    def __init__(self, img_rgb):
        self.img_rgb = img_rgb
        self.img_shape = img_rgb.shape
        
    # Menampilkan gambar yang di import
    def show_original_images(self):
        plt.imshow(self.img_rgb)
        plt.title('Original Image')
        plt.show()
        
    # Menghasilkan gambar blur
    def get_blur(self, level = 'low'):
        """
            level = very_low, low, med, high
        """
        ksize_x = self.img_shape[0]
        ksize_y = self.img_shape[1]
        if (level == 'very_low'):
            ksize = (ksize_x // int(ksize_x/2), ksize_y // int(ksize_y/2))
        elif (level == 'low'):
            ksize = (ksize_x // int(ksize_x/3), ksize_y // int(ksize_y/3))
        elif (level == 'med'):
            ksize = (ksize_x // int(ksize_x/4), ksize_y // int(ksize_y/4))
        elif (level == 'high'):
            ksize = (ksize_x // int(ksize_x/5), ksize_y // int(ksize_y/5))
        img_blur = cv2.blur(self.img_rgb, ksize)
        return img_blur
    
    # Menghasilkan gambar terpotong (CROP)
    def get_crop(self, level = 'low'):
        """
            level = very_low, low, med, high
        """
        shape = self.img_shape
        edges = [0,0]
        if (level == 'very_low'):
            scale = 50
        elif (level == 'low'):
            scale = 40 
        elif (level == 'med'):
            scale = 30
        elif (level == 'high'):
            scale = 15
        edges[0], edges[1] = shape[0] // scale, shape[1] // scale
        # 3 Dimensi
        if (len(self.img_shape) == 3):  
            img_crop = self.img_rgb.copy()[edges[0] : shape[0]-edges[0], edges[1] : shape[1]-edges[1], :]
        # 2 Dimensi
        elif (len(self.img_shape) == 2):  
            img_crop = self.img_rgb.copy()[edges[0] : shape[0]-edges[0], edges[1] : shape[1]-edges[1]]
        return img_crop

    # Menghasilkan gambar dengan intensitas yang berbeda
    def get_intensity(self, mode = 'glow'):
        """
            mode = glow, dark
            Source : https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
        """
        img_rgb = self.img_rgb
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        if (mode == 'glow'):
            value = 100               # PARAMETER
            h, s, v = cv2.split(hsv)
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
            hsv = cv2.merge((h,s,v))
            img_intensity = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif (mode == 'dark'):
            hsv[...,2] = hsv[...,2]*0.6
            img_intensity = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img_intensity
          
    # Menghasilkan gambar noise
    def get_noise(self, mode = 'gaussian'):
        """
            mode : gaussian, s&p, speckle
        """
        if (mode == 'speckle'):
            img_noise = skutil.random_noise(self.img_rgb, mode, var = 0.1)
        else:
            img_noise = skutil.random_noise(self.img_rgb, mode)
        img_noise = 255*(img_noise - img_noise.min())/(img_noise.max() - img_noise.min())
        img_noise = img_noise.astype('uint8')
        return img_noise
    
    # Menghasilkan gambar yang dirotasi
    def get_rotate(self, mode = 'left'):
        """
            mode : left, right
        """
        image = self.img_rgb
        if (mode == 'left'):
            angle = 10         #degree
        elif (mode == 'right'):
            angle = -10        #degree
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        img_rotate = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                                    flags=cv2.INTER_LINEAR)
        return img_rotate
    
    # Menghasilkan gambar yang ditranslasi keempat arah berdasarkan mode
    def get_translate(self, mode = 'up'):
        """
            mode : up, down, right, left
        """
        image = self.img_rgb
        shape = self.img_shape
        scale = [0,0]
        scale[0], scale[1] = shape[1] // 15, shape[0] // 10
        if (mode == 'up'):
            translation = (0, scale[1])
        elif (mode == 'down'):
            translation = (0, -scale[1])
        elif (mode == 'right'):
            translation = (-scale[0], 0)
        elif (mode == 'left'):
            translation = (scale[0], 0)
        transmat = sktrans.AffineTransform(translation = translation)
        img_trans = sktrans.warp(image, transmat, preserve_range=True).astype('uint8')
        return img_trans
    
    # Menghasilkan gambar yang di scale
    def get_scale(self, mode = 'small'):
        """
            mode : small, big
        """
        img = self.img_rgb
        shape = self.img_shape
        result = np.zeros(shape)
        if (mode == 'small'):
            img_scale = sktrans.rescale(img, 0.8, anti_aliasing = False)
            img_scale = 255*(img_scale - img_scale.min())/(img_scale.max() - img_scale.min())
            for i in range(img_scale.shape[0]):
                for j in range(img_scale.shape[1]):
                    result[i,j,:] = img_scale[i,j,:]
        elif (mode == 'big'):
            img_scale = sktrans.rescale(img, 1.2, anti_aliasing = False)
            img_scale = 255*(img_scale - img_scale.min())/(img_scale.max() - img_scale.min())
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    result[i,j,:] = img_scale[i,j,:]
        result = result.astype('uint8')
        return result
    
    # Menghasilkan gambar yang di geser (shear)
    def get_shear(self, mode = 'left'):
        """
            mode : left, right
        """
        img = self.img_rgb
        if (mode == 'left'):
            scale_shear = 0.3
        elif (mode == 'right'):
            scale_shear = -0.3
        transform = sktrans.AffineTransform(shear = scale_shear)
        img_shear =  sktrans.warp(img, transform, preserve_range=True).astype('uint8')
        return img_shear