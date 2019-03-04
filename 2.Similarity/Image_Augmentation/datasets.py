# Author : Bayu Aditya

import matplotlib.pyplot as plt

from .images import produce_image
from .aug_bbox import get_bbox_rotate
from .aug_bbox import get_bbox_shear

class image_augmentation():
    """
    Input : image 3 channels
    Output : dictionary dataset

    Produce 23 variational condition from 1 images
    """
    def __init__(self, img_rgb):
        self.img = img_rgb
        self._tools = produce_image(self.img)
        self.param = {'get_blur' : ['very_low', 'low', 'med', 'high'],
        'get_crop' : ['very_low', 'low', 'med', 'high'],
        'get_intensity' : ['glow', 'dark'],
        'get_noise' : ['gaussian', 's&p', 'speckle'],
        'get_rotate' : ['left', 'right'],
        'get_translate' : ['up', 'down', 'right', 'left'],
        'get_scale' : ['small', 'big'],
        'get_shear' : ['left', 'right']}

    def _create_initialization_dict_for_dataset(self):
        result_empty = {}
        parameter = self.param

        for subclass in parameter:
            result_empty.update({subclass : {}})
            for mode in parameter[subclass]:
                result_empty[subclass].update({mode : []})
        return result_empty

    def get_result(self):
        result = self._create_initialization_dict_for_dataset()
        for subclass in self.param.keys():
            for mode in self.param[subclass]:
                func = eval('self._tools.' + subclass)
                img_processs = func(mode)
                result[subclass][mode] = img_processs
        return result

    def view_result(self):
        parameter = self.param
        dict_aug = self.get_result()

        for subclass in parameter:
            for mode in parameter[subclass]:
                img_ = dict_aug[subclass][mode]
                title = subclass + ' : ' + mode
                plt.imshow(img_)
                plt.title(title)
                plt.show()